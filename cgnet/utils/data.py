from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Any, Dict
from tqdm import tqdm
from pathlib import Path

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import InMemoryDataset
except ModuleNotFoundError:
    raise ImportError("This function requires PyTorch Geometric to be installed.")

from ase import Atoms


class GraphData:
    """A graph data structure for crystal structure representation.
    
    This class encapsulates node features, edge connectivity, and additional
    attributes for representing crystal structures as graphs.
    """
    
    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_weights: Optional[np.ndarray] = None,
        id: Optional[str] = None,
        label: Optional[float] = None,
        cidxs: Optional[List[int]] = None,
        **kwargs,
    ):
        """Initialize GraphData object.
        
        Parameters
        ----------
        node_features : np.ndarray
            Node feature matrix with shape [num_nodes, num_node_features]
        edge_index : np.ndarray
            Graph connectivity in COO format with shape [2, num_edges]
        edge_features : np.ndarray, optional
            Edge feature matrix with shape [num_edges, num_edge_features]
        node_weights : np.ndarray, optional
            Node weight matrix with shape [num_nodes, ]
        id : str, optional
            The unique identifier for the datapoint
        label : float, optional
            The target value for the datapoint
        cidxs : List[int], optional
            The cluster center indexes for the datapoint
        **kwargs
            Additional attributes and their values
            
        Raises
        ------
        ValueError
            If input validation fails
        """
        self._validate_inputs(node_features, edge_index, edge_features)
        
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_weights = node_weights
        self.id = id
        self.label = label
        self.cidxs = cidxs
        self.kwargs = kwargs
        
        # Cache computed properties
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        self.num_edge_features = edge_features.shape[1] if edge_features is not None else 0

        # Set additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _validate_inputs(
        node_features: np.ndarray, 
        edge_index: np.ndarray, 
        edge_features: Optional[np.ndarray]
    ) -> None:
        """Validate input arrays for GraphData construction.
        
        Parameters
        ----------
        node_features : np.ndarray
            Node feature matrix to validate
        edge_index : np.ndarray
            Edge connectivity matrix to validate
        edge_features : np.ndarray, optional
            Edge feature matrix to validate
            
        Raises
        ------
        ValueError
            If any validation check fails
        """
        # Validate node_features
        if not isinstance(node_features, np.ndarray):
            raise ValueError("node_features must be np.ndarray")
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [num_nodes, num_node_features]")

        # Validate edge_index
        if not isinstance(edge_index, np.ndarray):
            raise ValueError("edge_index must be np.ndarray")
        if not np.issubdtype(edge_index.dtype, np.integer):
            raise ValueError("edge_index must contain integers")
        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        
        # Check edge_index bounds (only for non-empty arrays)
        if edge_index.size > 0 and np.max(edge_index) >= len(node_features):
            raise ValueError("edge_index contains invalid node indices")

        # Validate edge_features if provided
        if edge_features is not None:
            if not isinstance(edge_features, np.ndarray):
                raise ValueError("edge_features must be np.ndarray or None")
            if edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "edge_features first dimension must match edge_index second dimension"
                )

    def __repr__(self) -> str:
        """Return string representation of GraphData object."""
        info_parts = [
            f"num_nodes={self.num_nodes}",
            f"num_edges={self.num_edges}",
            f"num_node_features={self.num_node_features}",
        ]
        
        if self.edge_features is not None:
            info_parts.append(f"num_edge_features={self.num_edge_features}")
        if self.id is not None:
            info_parts.append(f"id={self.id}")
        if self.label is not None:
            info_parts.append(f"label={self.label}")
        if self.cidxs is not None:
            info_parts.append(f"cidxs={self.cidxs}")
            
        return f"{self.__class__.__name__}({', '.join(info_parts)})"

    def to_torch_geometric_data(self) -> Data:
        """Convert the graph to torch_geometric.data.Data format.

        Returns
        -------
        Data
            A torch geometric data object
        """
        # Convert numpy arrays to tensors with appropriate dtypes
        node_features = torch.tensor(self.node_features, dtype=torch.float32)
        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        
        edge_features = (
            torch.tensor(self.edge_features, dtype=torch.float32) 
            if self.edge_features is not None else None
        )
        node_weights = (
            torch.tensor(self.node_weights, dtype=torch.float32) 
            if self.node_weights is not None else None
        )
        label = (
            torch.tensor(self.label, dtype=torch.float32) 
            if self.label is not None else None
        )
        cidxs = (
            torch.tensor(self.cidxs, dtype=torch.long) 
            if self.cidxs is not None else None
        )
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            node_weights=node_weights,
            id=self.id,
            y=label,
            cidxs=cidxs,
        )

        return data


class CGNETDataset(InMemoryDataset):
    """CG-NET Dataset for PyTorch Geometric.
    
    A dataset class that follows PyTorch Geometric conventions for loading and processing
    crystal structure data into graph format using the CG-NET featurization approach.
    
    This class handles the conversion of crystal structures to graph representations,
    including validation, filtering, and preprocessing steps.
    """
    
    def __init__(
        self,
        root: str,
        ids: Optional[List[str]] = None,
        cidxs: Optional[List[List[int]]] = None,
        structures: Optional[List[Atoms]] = None,
        labels: Optional[List[float]] = None,
        featureizer = None,
        transform = None,
        pre_transform = None,
        pre_filter = None,
        force_reload: bool = False,
        filter_isolated_nodes: bool = True,
    ):
        """Initialize CGNETDataset.
        
        Parameters
        ----------
        root : str
            Root directory where the dataset should be saved
        ids : List[str], optional
            The unique identifier for each datapoint
        cidxs : List[List[int]], optional
            The cluster center indexes for each structure
        structures : List[Atoms], optional
            A list of ase Atoms objects
        labels : List[float], optional
            A list of target values
        featureizer : CGNETFeatureizer, optional
            The featurizer object to featurize the structure
        transform : callable, optional
            A function/transform that takes in a Data object and returns a transformed version
        pre_transform : callable, optional
            A function/transform that takes in a Data object and returns a transformed version
        pre_filter : callable, optional
            A function that takes in a Data object and returns a boolean value
        force_reload : bool, default False
            Whether to re-process the dataset even if processed files exist
        filter_isolated_nodes : bool, default True
            Whether to filter out graphs containing isolated nodes
        """
        # Store and validate raw data
        self._raw_ids = self._validate_list_input(ids, "ids")
        self._raw_cidxs = self._validate_list_input(cidxs, "cidxs")
        self._raw_structures = self._validate_list_input(structures, "structures")
        self._raw_labels = self._validate_list_input(labels, "labels")
        
        if self._raw_structures and not featureizer:
            raise ValueError("Featurizer must be provided when structures are given")
            
        self.featureizer = featureizer
        self.filter_isolated_nodes = filter_isolated_nodes
        
        # Initialize parent class
        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
            force_reload=force_reload
        )
        
        # Load processed data
        if os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])

    @staticmethod
    def _validate_list_input(input_list: Optional[List], name: str) -> List:
        """Validate and normalize list inputs."""
        if input_list is None:
            return []
        if not isinstance(input_list, list):
            raise ValueError(f"{name} must be a list or None")
        return input_list

    @property
    def raw_file_names(self) -> List[str]:
        """Names of raw files needed for processing.
        
        Since we're providing data directly, we don't need raw files.
        """
        return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """Names of processed files that need to exist to skip processing."""
        return ['data.pt']
    
    def download(self) -> None:
        """Download the dataset to the raw_dir folder.
        
        Since we're providing data directly, this method does nothing.
        """
        pass
    
    def process(self) -> None:
        """Process the dataset to the processed_dir folder.
        
        This method converts raw crystal structures into PyG Data objects,
        applies filtering, and saves the processed data.
        """
        if not self._raw_structures:
            self._process_empty_dataset()
            return
            
        if not self.featureizer:
            raise ValueError("Featurizer must be provided for processing")
        
        print(f"Processing {len(self._raw_structures)} structures...")
        
        data_list, filtered_graphs = self._process_structures()
        
        # Save processed data (PyG will automatically apply pre_filter and pre_transform)
        self.save(data_list, self.processed_paths[0])
        
        # Save filtering information
        if filtered_graphs:
            self._save_filtered_graphs_info(filtered_graphs)
        
        self._print_processing_summary(data_list, filtered_graphs)
    
    def _process_empty_dataset(self) -> None:
        """Handle processing when no structures are provided."""
        data_list = []
        self.save(data_list, self.processed_paths[0])
        print("No structures provided. Created empty dataset.")
    
    def _process_structures(self) -> Tuple[List[Data], List[Dict[str, Any]]]:
        """Process structures into graph data objects.
        
        Returns
        -------
        Tuple[List[Data], List[Dict]]
            Valid data objects and information about filtered graphs
        """
        data_list = []
        filtered_graphs = []
        
        for i in tqdm(range(len(self._raw_structures)), desc="Processing graphs"):
            try:
                graph_data = self._process_single_structure(i)
                
                if graph_data is None:
                    continue
                    
                graph, filter_info = graph_data
                
                if filter_info:
                    filtered_graphs.append(filter_info)
                    print(f"Filtered out graph {filter_info['id']}: {filter_info['reason']} "
                          f"(nodes: {filter_info['num_nodes']}, edges: {filter_info['num_edges']})")
                else:
                    data_list.append(graph)
                    
            except Exception as e:
                error_info = self._create_error_info(i, str(e))
                filtered_graphs.append(error_info)
                print(f"Error processing {error_info['id']}: {e}")
        
        return data_list, filtered_graphs
    
    def _process_single_structure(self, index: int) -> Optional[Tuple[Data, Optional[Dict[str, Any]]]]:
        """Process a single structure into a graph.
        
        Parameters
        ----------
        index : int
            Index of the structure to process
            
        Returns
        -------
        Optional[Tuple[Data, Optional[Dict]]]
            Graph data and optional filter information, or None if processing failed
        """
        # Get structure data
        id_val = self._get_safe_value(self._raw_ids, index, f"structure_{index}")
        label_val = self._get_safe_value(self._raw_labels, index, 0.0)
        struct = self._raw_structures[index]
        cidx_val = self._get_safe_value(self._raw_cidxs, index, [0])
        
        # Featurize structure
        graph = self.featureizer._featurize(
            struct, 
            id=id_val, 
            label=label_val, 
            cidxs=cidx_val
        )
        
        # Check validity
        filter_reason = self._check_graph_validity(graph)
        filter_info = None
        
        if filter_reason:
            filter_info = {
                'id': id_val,
                'label': label_val,
                'num_nodes': graph.x.size(0),
                'num_edges': graph.edge_index.size(1),
                'num_atoms': len(struct),
                'cidxs': cidx_val,
                'reason': filter_reason
            }
        
        return graph, filter_info
    
    @staticmethod
    def _get_safe_value(data_list: List, index: int, default):
        """Safely get value from list with default fallback."""
        return data_list[index] if index < len(data_list) else default
    
    def _create_error_info(self, index: int, error_msg: str) -> Dict[str, Any]:
        """Create error information dictionary for failed processing."""
        id_val = self._get_safe_value(self._raw_ids, index, f"structure_{index}")
        label_val = self._get_safe_value(self._raw_labels, index, 0.0)
        struct = self._raw_structures[index]
        cidx_val = self._get_safe_value(self._raw_cidxs, index, [0])
        
        return {
            'id': id_val,
            'label': label_val,
            'num_nodes': 0,
            'num_edges': 0,
            'num_atoms': len(struct),
            'cidxs': cidx_val,
            'reason': f'processing_error: {error_msg}'
        }
    

    
    def _check_graph_validity(self, graph: Data) -> Optional[str]:
        """Check if a graph is valid based on filtering criteria.
        
        Parameters
        ----------
        graph : Data
            Graph to check
            
        Returns
        -------
        Optional[str]
            Reason for filtering, or None if graph is valid
        """
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)
        
        # Check for no nodes
        if num_nodes == 0:
            return "no_nodes"
        
        # Check for no edges
        if num_edges == 0:
            return "no_edges"
        
        # Check for isolated nodes if filtering is enabled
        if self.filter_isolated_nodes:
            edge_nodes = torch.unique(graph.edge_index.flatten())
            connected_nodes = len(edge_nodes)
            
            if connected_nodes < num_nodes:
                isolated_count = num_nodes - connected_nodes
                return f"isolated_nodes: {isolated_count}/{num_nodes}"
        
        return None
    
    def _save_filtered_graphs_info(self, filtered_graphs: List[Dict[str, Any]]) -> None:
        """Save information about filtered graphs to files.
        
        Parameters
        ----------
        filtered_graphs : List[Dict]
            List of dictionaries containing information about filtered graphs
        """
        # Ensure processed directory exists
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed CSV
        df = pd.DataFrame(filtered_graphs)
        csv_path = processed_dir / "filtered_graphs.csv"
        df.to_csv(csv_path, index=False)
        
        # Save summary
        self._save_filtering_summary(processed_dir, filtered_graphs)
    
    def _save_filtering_summary(self, processed_dir: Path, filtered_graphs: List[Dict[str, Any]]) -> None:
        """Save filtering summary to text file."""
        summary_path = processed_dir / "filtering_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("Graph Filtering Summary\n")
            f.write("=====================\n\n")
            f.write(f"Total filtered graphs: {len(filtered_graphs)}\n")
            f.write(f"Filtering timestamp: {pd.Timestamp.now()}\n\n")
            
            # Count by reason
            reason_counts = {}
            for graph in filtered_graphs:
                reason = graph.get('reason', 'unknown')
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            f.write("Filtering reasons:\n")
            for reason, count in reason_counts.items():
                f.write(f"  - {reason}: {count}\n")
            
            # Write statistics
            self._write_filtering_statistics(f, filtered_graphs)
    
    def _write_filtering_statistics(self, file, filtered_graphs: List[Dict[str, Any]]) -> None:
        """Write filtering statistics to file."""
        node_counts = [g.get('num_nodes', 0) for g in filtered_graphs]
        edge_counts = [g.get('num_edges', 0) for g in filtered_graphs]
        
        if node_counts and any(count > 0 for count in node_counts):
            valid_nodes = [c for c in node_counts if c > 0]
            file.write(f"\nNode count statistics for filtered graphs:\n")
            file.write(f"  - Min nodes: {min(valid_nodes)}\n")
            file.write(f"  - Max nodes: {max(valid_nodes)}\n")
            file.write(f"  - Mean nodes: {sum(valid_nodes)/len(valid_nodes):.1f}\n")
        
        if edge_counts and any(count > 0 for count in edge_counts):
            valid_edges = [c for c in edge_counts if c > 0]
            file.write(f"\nEdge count statistics for filtered graphs:\n")
            file.write(f"  - Min edges: {min(valid_edges)}\n")
            file.write(f"  - Max edges: {max(valid_edges)}\n")
            file.write(f"  - Mean edges: {sum(valid_edges)/len(valid_edges):.1f}\n")
    
    def _print_processing_summary(self, data_list: List[Data], filtered_graphs: List[Dict[str, Any]]) -> None:
        """Print processing summary."""
        total_original = len(self._raw_structures)
        total_final = len(data_list)
        total_filtered = len(filtered_graphs)
        
        print(f"\nDataset processing complete:")
        print(f"  - Original structures: {total_original}")
        print(f"  - Final valid graphs: {total_final}")
        print(f"  - Filtered problematic graphs: {total_filtered}")
        print(f"  - Success rate: {total_final/max(total_original, 1)*100:.1f}%")
        print(f"Saved {total_final} processed graphs to {self.processed_paths[0]}")
        
        if filtered_graphs:
            print(f"Filtered graphs information saved to {self.processed_dir}/filtered_graphs.csv")
    

