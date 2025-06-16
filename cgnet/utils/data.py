from __future__ import annotations

import os
import json
import numpy as np
from typing import Tuple
from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from itertools import repeat

try:
    import torch
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Batch
    from torch_geometric.data import InMemoryDataset
except ModuleNotFoundError:
    raise ImportError("This function requires PyTorch Geometric to be installed.")

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


class GraphData:
    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: np.ndarray | None = None,
        node_weights: np.ndarray | None = None,
        id: str = None,
        label: float = None,
        cidxs: list[int] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_features: np.ndarray
            Node feature matrix with shape [num_nodes, num_node_features]
        edge_index: np.ndarray, dtype int
            Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
            Edge feature matrix with shape [num_edges, num_edge_features]
        node_weights: np.ndarray, optional (default None)
            Node weight matrix with shape [num_nodes, ]
        id: str, optional (default None)
            The unique identifier for the datapoint.
        label: float, optional (default None)
            The target value for the datapoint.
        cidxs: list[int], optional (default None)
            The cluster center indexes for the datapoint.
        kwargs: optional
            Additional attributes and their values
        """

        if isinstance(node_features, np.ndarray) is False:
            raise ValueError("node_features must be np.ndarray.")
        elif node_features.ndim != 2:
            raise ValueError("The shape of node_features is [num_nodes, num_node_features].")

        if isinstance(edge_index, np.ndarray) is False:
            raise ValueError("edge_index must be np.ndarray.")
        elif issubclass(edge_index.dtype.type, np.integer) is False:
            raise ValueError("edge_index.dtype must contains integers.")
        elif edge_index.shape[0] != 2:
            raise ValueError("The shape of edge_index is [2, num_edges].")

        # np.max() method works only for a non-empty array, so size of the array should be non-zero
        elif (edge_index.size != 0) and (np.max(edge_index) >= len(node_features)):
            raise ValueError("edge_index contains the invalid node number.")

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray) is False:
                raise ValueError("edge_features must be np.ndarray or None.")
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "The first dimension of edge_features must be the same as the second dimension of edge_index."
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_weights = node_weights
        self.id = id
        self.label = label
        self.cidxs = cidxs
        self.kwargs = kwargs
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        info = [
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"num_node_features={self.num_node_features}",
        ]
        if self.edge_features is not None:
            info.append(f"num_edge_features={self.num_edge_features}")
        if self.id is not None:
            info.append(f"id={self.id}")
        if self.label is not None:
            info.append(f"label={self.label}")
        if self.cidxs is not None:
            info.append(f"cidxs={self.cidxs}")
        info_str = ", ".join(info)
        return f"{self.__class__.__name__}({info_str})"

    def to_torch_geometric_data(self):
        """
        Convert the graph to torch_geometric.data.Data format.

        Returns
        -------
        data: torch_geometric.data.Data
            A torch geometric data object.
        """
        node_features = torch.tensor(self.node_features, dtype=torch.float)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_features = torch.tensor(self.edge_features, dtype=torch.float) if self.edge_features is not None else None
        node_weights = torch.tensor(self.node_weights, dtype=torch.float) if self.node_weights is not None else None
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            node_weights=node_weights,
            id=self.id,
            y=torch.tensor(self.label, dtype=torch.float) if self.label is not None else None,
            cidxs=torch.tensor(self.cidxs, dtype=torch.long) if self.cidxs is not None else None,
        )

        return data


class CGNETDataset(InMemoryDataset):
    """
    CG-NET Dataset for PyTorch Geometric
    
    A dataset class that follows PyTorch Geometric conventions for loading and processing
    crystal structure data into graph format using the CG-NET featurization approach.
    """
    
    def __init__(
        self,
        root: str,
        ids: list = None,
        cidxs: list = None,
        structures: list = None,
        labels: list = None,
        featureizer = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_parallel: bool = False,
        force_reload: bool = False,
        filter_isolated_nodes: bool = True,
    ):
        """
        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        ids : list, optional
            The unique identifier for each datapoint.
        cidxs : list, optional
            The cluster center indexes for each structure.
        structures : list, optional
            A list of ase Atoms objects.
        labels : list, optional
            A list of target values.
        featureizer : CGNETFeatureizer, optional
            The featurizer object to featurize the structure.
        transform : callable, optional
            A function/transform that takes in an torch_geometric.data.Data object 
            and returns a transformed version.
        pre_transform : callable, optional
            A function/transform that takes in an torch_geometric.data.Data object 
            and returns a transformed version.
        pre_filter : callable, optional
            A function that takes in an torch_geometric.data.Data object and returns 
            a boolean value, indicating whether the data object should be included in the final dataset.
        use_parallel : bool, optional (default False)
            Whether to use parallel processing for graph generation.
        force_reload : bool, optional (default False)
            Whether to re-process the dataset even if processed files exist.
        filter_isolated_nodes : bool, optional (default True)
            Whether to filter out graphs containing isolated nodes (nodes with no edges).
            Set to False if isolated nodes are meaningful in your application.
        """
        # Store raw data
        self._raw_ids = ids or []
        self._raw_cidxs = cidxs or []
        self._raw_structures = structures or []
        self._raw_labels = labels or []
        self.featureizer = featureizer
        self.use_parallel = use_parallel
        self.filter_isolated_nodes = filter_isolated_nodes
        
        # Initialize parent class
        super(CGNETDataset, self).__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter
        )
        
        # Load processed data if not forced to reload
        if not force_reload:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """
        The name of the files in the raw_dir folder that need to be found in order 
        to skip the download.
        """
        # Since we're providing data directly, we don't need raw files
        return []
    
    @property
    def processed_file_names(self):
        """
        The name of the files in the processed_dir folder that need to be found 
        in order to skip the processing.
        """
        return ['data.pt']
    
    def download(self):
        """
        Downloads the dataset to the raw_dir folder.
        Since we're providing data directly, this method does nothing.
        """
        pass
    
    def process(self):
        """
        Processes the dataset to the processed_dir folder.
        This method converts raw crystal structures into PyG Data objects.
        Filters out graphs with no edges and isolated node graphs.
        """
        if not self.featureizer:
            raise ValueError("Featurizer must be provided for processing")
        
        if not self._raw_structures:
            # No data to process, create empty dataset
            data_list = []
            filtered_graphs = []
        else:
            print(f"Processing {len(self._raw_structures)} structures...")
            
            # Process structures into graphs
            data_list = []
            filtered_graphs = []  # Store information about filtered graphs
            
            if self.use_parallel:
                # Import the parallel processing function
                from .featureizer import _generate_graphs_chunk
                
                # Parallel processing
                chunk_size = 1000
                tasks = []
                for start_idx in range(0, len(self._raw_structures), chunk_size):
                    end_idx = min(start_idx + chunk_size, len(self._raw_structures))
                    tasks.append((
                        start_idx, end_idx, 
                        self._raw_ids, self._raw_labels, 
                        self._raw_structures, self._raw_cidxs, 
                        self.featureizer
                    ))

                print(f"Processing {len(tasks)} chunks in parallel...")
                with Pool(processes=mp.cpu_count()) as pool:
                    chunk_results = pool.starmap(_generate_graphs_chunk, tasks)
                
                # Flatten results and filter problematic graphs
                for chunk in chunk_results:
                    for graph in chunk:
                        filter_reason = self._check_graph_validity(graph)
                        if filter_reason:
                            filtered_graphs.append({
                                'id': graph.id,
                                'label': graph.y.item() if graph.y is not None else None,
                                'num_nodes': graph.x.size(0),
                                'num_edges': graph.edge_index.size(1),
                                'cidxs': graph.cidxs.tolist() if graph.cidxs is not None else None,
                                'reason': filter_reason
                            })
                        else:
                            data_list.append(graph)
            else:
                # Sequential processing
                for i in tqdm(range(len(self._raw_structures)), desc="Processing graphs"):
                    id_val = self._raw_ids[i] if i < len(self._raw_ids) else f"structure_{i}"
                    label_val = self._raw_labels[i] if i < len(self._raw_labels) else 0.0
                    struct = self._raw_structures[i]
                    cidx_val = self._raw_cidxs[i] if i < len(self._raw_cidxs) else [0]
                    
                    try:
                        graph = self.featureizer._featurize(
                            struct, 
                            id=id_val, 
                            label=label_val, 
                            cidxs=cidx_val
                        )
                        
                        # Check graph validity
                        filter_reason = self._check_graph_validity(graph)
                        if filter_reason:
                            filtered_graphs.append({
                                'id': id_val,
                                'label': label_val,
                                'num_nodes': graph.x.size(0),
                                'num_edges': graph.edge_index.size(1),
                                'num_atoms': len(struct),
                                'cidxs': cidx_val,
                                'reason': filter_reason
                            })
                            print(f"  ⚠️  Filtered out graph {id_val}: {filter_reason} (nodes: {graph.x.size(0)}, edges: {graph.edge_index.size(1)})")
                        else:
                            data_list.append(graph)
                            
                    except Exception as e:
                        print(f"  ❌ Error processing {id_val}: {e}")
                        filtered_graphs.append({
                            'id': id_val,
                            'label': label_val,
                            'num_nodes': 0,
                            'num_edges': 0,
                            'num_atoms': len(struct),
                            'cidxs': cidx_val,
                            'reason': f'processing_error: {str(e)}'
                        })
                        continue
            
            # Save information about filtered graphs
            if filtered_graphs:
                self._save_filtered_graphs_info(filtered_graphs)
                print(f"Filtered out {len(filtered_graphs)} problematic graphs")
                print(f"Filtered graphs information saved to {self.processed_dir}/filtered_graphs.csv")
        
        # Apply pre-filtering and pre-transformation
        if self.pre_filter is not None:
            original_count = len(data_list)
            data_list = [data for data in data_list if self.pre_filter(data)]
            filtered_count = original_count - len(data_list)
            if filtered_count > 0:
                print(f"Pre-filter removed {filtered_count} additional graphs")

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save processed data using PyG's collate function
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        total_original = len(self._raw_structures) if self._raw_structures else 0
        total_final = len(data_list)
        total_filtered = len(filtered_graphs) if filtered_graphs else 0
        
        print(f"Dataset processing complete:")
        print(f"  - Original structures: {total_original}")
        print(f"  - Final valid graphs: {total_final}")
        print(f"  - Filtered problematic graphs: {total_filtered}")
        print(f"  - Success rate: {total_final/max(total_original, 1)*100:.1f}%")
        print(f"Saved {total_final} processed graphs to {self.processed_paths[0]}")
    
    def _check_graph_validity(self, graph):
        """
        Check if a graph is valid based on filtering criteria
        
        Parameters
        ----------
        graph : torch_geometric.data.Data
            Graph to check
            
        Returns
        -------
        str or None
            Reason for filtering, or None if graph is valid
        """
        # Always check for no edges - graphs with no edges are generally problematic
        if graph.edge_index.size(1) == 0:
            return "no_edges"
        
        # Always check for no nodes
        num_nodes = graph.x.size(0)
        if num_nodes == 0:
            return "no_nodes"
        
        # Check for isolated nodes only if filtering is enabled
        if self.filter_isolated_nodes:
            # Get all nodes that appear in edges
            edge_nodes = torch.unique(graph.edge_index.flatten())
            connected_nodes = len(edge_nodes)
            
            # If there are nodes not appearing in any edge, they are isolated
            if connected_nodes < num_nodes:
                isolated_count = num_nodes - connected_nodes
                return f"isolated_nodes: {isolated_count}/{num_nodes}"
        
        # Graph is valid
        return None
    
    def _save_filtered_graphs_info(self, filtered_graphs):
        """
        Save information about filtered graphs to a CSV file
        
        Parameters
        ----------
        filtered_graphs : list
            List of dictionaries containing information about filtered graphs
        """
        import pandas as pd
        import os
        
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(filtered_graphs)
        csv_path = os.path.join(self.processed_dir, "filtered_graphs.csv")
        df.to_csv(csv_path, index=False)
        
        # Also save a summary
        summary_path = os.path.join(self.processed_dir, "filtering_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Graph Filtering Summary\n")
            f.write(f"=====================\n\n")
            f.write(f"Total filtered graphs: {len(filtered_graphs)}\n")
            f.write(f"Filtering timestamp: {pd.Timestamp.now()}\n\n")
            
            # Count by reason
            if filtered_graphs:
                reason_counts = {}
                for graph in filtered_graphs:
                    reason = graph.get('reason', 'unknown')
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                
                f.write("Filtering reasons:\n")
                for reason, count in reason_counts.items():
                    f.write(f"  - {reason}: {count}\n")
                    
                # Statistics
                node_counts = [g['num_nodes'] for g in filtered_graphs if 'num_nodes' in g]
                edge_counts = [g['num_edges'] for g in filtered_graphs if 'num_edges' in g]
                
                if node_counts:
                    f.write(f"\nNode count statistics for filtered graphs:\n")
                    f.write(f"  - Min nodes: {min(node_counts)}\n")
                    f.write(f"  - Max nodes: {max(node_counts)}\n")
                    f.write(f"  - Mean nodes: {sum(node_counts)/len(node_counts):.1f}\n")
                
                if edge_counts:
                    f.write(f"\nEdge count statistics for filtered graphs:\n")
                    f.write(f"  - Min edges: {min(edge_counts)}\n")
                    f.write(f"  - Max edges: {max(edge_counts)}\n")
                    f.write(f"  - Mean edges: {sum(edge_counts)/len(edge_counts):.1f}\n")
    
    @classmethod
    def from_data_lists(
        cls,
        root: str,
        ids: list,
        cidxs: list,
        structures: list,
        labels: list,
        featureizer,
        **kwargs
    ):
        """
        Create a CGNETDataset from lists of data.
        
        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved.
        ids : list
            The unique identifier for each datapoint.
        cidxs : list
            The cluster center indexes for each structure.
        structures : list
            A list of ase Atoms objects.
        labels : list
            A list of target values.
        featureizer : CGNETFeatureizer
            The featurizer object to featurize the structure.
        **kwargs
            Additional arguments passed to the constructor, including:
            - filter_isolated_nodes : bool, optional (default True)
              Whether to filter out graphs containing isolated nodes.
            
        Returns
        -------
        CGNETDataset
            The created dataset.
        """
        return cls(
            root=root,
            ids=ids,
            cidxs=cidxs,
            structures=structures,
            labels=labels,
            featureizer=featureizer,
            **kwargs
        )
    
    @classmethod
    def from_cache(cls, root: str, **kwargs):
        """
        Create a CGNETDataset from cached processed data.
        
        Parameters
        ----------
        root : str
            Root directory where the processed dataset is saved.
        **kwargs
            Additional arguments passed to the constructor, including:
            - filter_isolated_nodes : bool, optional (default True)
              Whether to filter out graphs containing isolated nodes.
            Note: This parameter only affects reprocessing when force_reload=True.
            
        Returns
        -------
        CGNETDataset
            The loaded dataset.
        """
        return cls(root=root, **kwargs)
