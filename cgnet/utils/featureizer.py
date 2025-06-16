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

from .data import GraphData


# Global function for multiprocessing (moved outside the class)
def _generate_graphs_chunk(start_idx, end_idx, ids, labels, structures, cidxs, featureizer):
    """
    Generate graphs for a chunk of structures.
    This function needs to be at module level for multiprocessing.
    
    Parameters
    ----------
    start_idx : int
        Starting index of the chunk
    end_idx : int
        Ending index of the chunk
    ids : list
        List of structure IDs
    labels : list
        List of labels/target values
    structures : list
        List of ASE Atoms objects
    cidxs : list
        List of cluster center indices
    featureizer : CGNETFeatureizer
        The featurizer instance to use
        
    Returns
    -------
    list
        List of generated graph objects
    """
    chunk_graphs = []
    for i in range(start_idx, end_idx):
        id = ids[i]
        label = labels[i]
        struct = structures[i]
        cidx = cidxs[i]
        graph = featureizer._featurize(struct, id=id, label=label, cidxs=cidx)
        chunk_graphs.append(graph)
    return chunk_graphs


class CGNETFeatureizer:
    def __init__(self, 
                 method: str = "CR",
                 neighbor_radius: float = 8.0,
                 max_neighbors: int = 12,
                 cluster_radius: float = 6.0,
                 neighbor_depth: int = 2,
                 max_cluster_nodes: int = None,
                 neighbor_mult: float = 1.15,
                 tag: int = 2,
                 step: float = 0.2,
                 with_pseudonodes: bool = True,
                 max_distance_factor: float = 2.0,
                 small_lattice_threshold: float = 5.0,
                 enable_smart_images: bool = True,
                 conservative_small_lattice: bool = False,
                 json_path: str = None,
                 json_file: str = "atom_init.json",
                 **kwargs):
        """
        Parameters
        ----------
        method: str, optional (default "radius")
            The method to featurize the structure. Currently only "CR" and "nth-NN" are supported.
        neighbor_radius: float, optional (default=8.0)
            The cutoff radius for the neighbor list.
        max_neighbors: int, optional (default=12)
            The maximum number of neighbors for each atom.
        cluster_radius: float, optional (default=6.0)
            The cutoff radius for the cluster.
        neighbor_depth: int or None, optional (default=2)
            The depth of the neighbor search. If None, the depth is not limited
        max_cluster_nodes: int or None, optional (default=None)
            The maximum number of cluster nodes. If None, the number is not limited.
        neighbor_mult: float or None, optional (default=1.15)
            The multiplicative factor for the neighbor radius. If None, the factor is not applied.
        tag: int, optional (default=2)
            The tag for the cluster center.
        step: float, optional (default=0.2)
            The step size for the Gaussian filter.
        with_pseudonodes: bool, optional (default=True)
            Whether to include pseudonodes in the graph.
        max_distance_factor: float, optional (default=2.0)
            The maximum distance factor for the neighbor search.
        small_lattice_threshold: float, optional (default=5.0)
            The threshold for small lattice detection.
        enable_smart_images: bool, optional (default=True)
            Whether to enable smart images.
        conservative_small_lattice: bool, optional (default=False)
            Whether to use conservative small lattice strategy.
        json_path: str, optional (default=None)
            The path to the JSON file containing the atom features. If None, uses the package directory.
        json_file: str, optional (default="atom_init.json")
            The name of the JSON file containing the atom features.
        """
        self.method = method
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.cluster_radius = cluster_radius
        self.neighbor_depth = neighbor_depth
        self.max_cluster_nodes = max_cluster_nodes
        self.neighbor_mult = neighbor_mult
        self.tag = tag
        self.step = step
        self.with_pseudonodes = with_pseudonodes
        self.max_distance_factor = max_distance_factor
        self.small_lattice_threshold = small_lattice_threshold
        self.enable_smart_images = enable_smart_images
        self.conservative_small_lattice = conservative_small_lattice

        # Set default json_path to current package directory if not provided
        if json_path is None:
            self.json_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.json_path = json_path
        self.json_file = json_file

        if method not in ["CR", "nth-NN"]:
            raise ValueError("Method must be either 'CR' or 'nth-NN'.")

        if self.method == "CR":
            if cluster_radius is None:
                raise ValueError("cluster_radius must be specified for CR method.")
            print(f"Using cluster radius method with cluster radius {self.cluster_radius} and max_cluster_nodes {self.max_cluster_nodes} to construct the graph dataset.")
        elif self.method == "nth-NN":
            if neighbor_depth is None:
                raise ValueError("neighbor_depth must be specified for nth-NN method.")
            print(f"Using nth-NN method with neighbor depth {self.neighbor_depth} and max_cluster_nodes {self.max_cluster_nodes} to construct the graph dataset.")

        # load atom_init.json
        atom_features_path = os.path.join(self.json_path, self.json_file)
        with open(atom_features_path, "r") as f:
            atom_init_json = json.load(f)

        self.atom_features = {
            int(key): np.array(value, dtype=np.float32)
            for key, value in atom_init_json.items()
        }
        self.valid_atom_number = set(self.atom_features.keys())
    
    def _detect_small_lattice(self, atom_object, threshold=None):
        """
        Detect if the structure is a small lattice
        
        Parameters
        ----------
        atom_object : ase.Atoms
            Atomic structure object
        threshold : float, optional
            Small lattice threshold (Angstrom)
        
        Returns
        -------
        bool
            Whether the structure is a small lattice
        """
        if threshold is None:
            threshold = self.small_lattice_threshold
            
        cell = atom_object.get_cell()
        cell_lengths = np.linalg.norm(cell, axis=1)
        # exclude zero length dimension (e.g. molecular system)
        valid_lengths = cell_lengths[cell_lengths > 1e-6]
        
        if len(valid_lengths) == 0:
            return False  # molecular system
        
        return np.any(valid_lengths < threshold)
    
    def _is_valid_distance(self, offset, atom_object, max_distance_factor=None):
        """
        Check if the offset is in a reasonable range to prevent small lattice infinite loop
        
        Parameters
        ----------
        offset : np.ndarray or list
            Periodic offset vector
        atom_object : ase.Atoms
            Atomic structure object
        max_distance_factor : float, optional
            Maximum distance factor, relative to the cell size
        
        Returns
        -------
        bool
            Whether the distance is valid
        """
        if max_distance_factor is None:
            max_distance_factor = self.max_distance_factor
            
        offset = np.array(offset)
        
        # zero offset is always valid
        if np.allclose(offset, [0, 0, 0]):
            return True
        
        # calculate the actual spatial distance
        cell = atom_object.get_cell()
        real_offset = np.dot(offset, cell)
        distance = np.linalg.norm(real_offset)
        
        # calculate the characteristic size of the cell
        cell_lengths = np.linalg.norm(cell, axis=1)
        valid_lengths = cell_lengths[cell_lengths > 1e-6]
        
        if len(valid_lengths) == 0:
            # molecular system, use the neighbor search radius
            distance_threshold = self.neighbor_radius * 2.0
        else:
            min_cell_length = np.min(valid_lengths)
            # distance threshold: consider the cell size and neighbor search radius
            distance_threshold = max(
                min_cell_length * max_distance_factor,
                self.neighbor_radius,
                10.0  # absolute minimum threshold (Angstrom)
            )
        
        return distance < distance_threshold
    
    def _find_all_connected_tag_nodes(self, nl, start_idx, atom_object, visited_tag_nodes=None):
        """
        Recursively search all tag nodes connected to the starting node
        
        Parameters
        ----------
        nl : NeighborList
            Neighbor list object
        start_idx : int
            Starting node index
        atom_object : ase.Atoms
            Atomic structure object
        visited_tag_nodes : set, optional
            Visited tag nodes set, avoid duplicate search
        
        Returns
        -------
        tag_nodes : list[tuple]
            List of all connected tag nodes: [(node index, offset)]
        """
        if visited_tag_nodes is None:
            visited_tag_nodes = set()
        
        tag_nodes = []
        queue = [(start_idx, [0, 0, 0])]
        local_visited = set()
        
        while queue:
            node_idx, offset = queue.pop(0)
            node_key = (node_idx, tuple(offset))
            
            if node_key in local_visited:
                continue
            local_visited.add(node_key)
            
            # if the node is a tag node and not visited, add to the result
            if atom_object[node_idx].tag == self.tag and node_key not in visited_tag_nodes:
                tag_nodes.append((node_idx, offset))
                visited_tag_nodes.add(node_key)
                
                # continue searching the tag nodes in the neighbor of the tag node
                try:
                    i_indices, i_offsets = nl.get_neighbors(node_idx)
                    for i, i_indice in enumerate(i_indices):
                        new_offset = np.array(offset) + np.array(i_offsets[i])
                        new_key = (i_indice, tuple(new_offset))
                        
                        if (atom_object[i_indice].tag == self.tag and 
                            new_key not in visited_tag_nodes and 
                            new_key not in local_visited and
                            self._is_valid_distance(new_offset, atom_object)):
                            queue.append((i_indice, new_offset.tolist()))
                except:
                    # handle the exception of getting neighbors
                    continue
        
        return tag_nodes
    
    def _find_tag_neighbors(self, nl, node_idx, offset, atom_object):
        """
        Find all tag neighbors of the specified node
        
        Parameters
        ----------
        nl : NeighborList
            Neighbor list object
        node_idx : int
            Node index
        offset : list
            Periodic offset of the node
        atom_object : ase.Atoms
            Atomic structure object
        
        Returns
        -------
        tag_neighbors : list[tuple]
            List of tag neighbors: [(node index, offset)]
        """
        tag_neighbors = []
        
        try:
            i_indices, i_offsets = nl.get_neighbors(node_idx)
            
            for i, i_indice in enumerate(i_indices):
                if atom_object[i_indice].tag == self.tag:
                    new_offset = np.array(offset) + np.array(i_offsets[i])
                    if self._is_valid_distance(new_offset, atom_object):
                        tag_neighbors.append((i_indice, new_offset.tolist()))
        except:
            # handle the exception of getting neighbors
            pass
        
        return tag_neighbors
    
    def _should_include_image(self, node_idx, offset, atom_object, visited_nodes):
        """
        Determine if the image atom should be included
        
        Parameters
        ----------
        node_idx : int
            Node index
        offset : list
            Periodic offset
        atom_object : ase.Atoms
            Atomic structure object
        visited_nodes : set
            Visited nodes set
        
        Returns
        -------
        bool
            Whether to include the image atom
        """
        if not self.enable_smart_images:
            return True
        
        node_key = (node_idx, tuple(offset))
        
        # check if the node is visited
        if node_key in visited_nodes:
            return False
        
        # check if the distance is valid
        if not self._is_valid_distance(np.array(offset), atom_object):
            return False
        
        # for tag nodes, more relaxed image strategy
        if atom_object[node_idx].tag == self.tag:
            return True
        
        # for non-tag nodes, check if the distance is reasonable
        if not np.allclose(offset, [0, 0, 0]):
            cell = atom_object.get_cell()
            real_offset = np.dot(offset, cell)
            distance = np.linalg.norm(real_offset)
            
            # more relaxed distance limit
            max_distance = self.neighbor_radius * 1.5
            return distance <= max_distance
        
        return True
    
    def _build_conservative_cluster(self, nl, cidx, atom_object):
        """
        Build a conservative cluster (for small lattice structures)
        
        Parameters
        ----------
        nl : NeighborList
            Neighbor list object
        cidx : int
            Cluster center index
        atom_object : ase.Atoms
            Atomic structure object
        
        Returns
        -------
        cluster_nodes : list[tuple]
            List of cluster nodes: [(node index, offset)]
        """
        cluster_nodes = []
        visited_nodes = set()
        
        # add the center node
        center_key = (cidx, (0, 0, 0))
        cluster_nodes.append((cidx, [0, 0, 0]))
        visited_nodes.add(center_key)
        
        # conservative BFS search, strictly limit the distance
        queue = [(cidx, [0, 0, 0], 0)]
        max_depth = min(self.neighbor_depth, 2)  # limit the maximum depth
        
        while queue:
            node_idx, offset, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            try:
                i_indices, i_offsets = nl.get_neighbors(node_idx)
                for i, i_indice in enumerate(i_indices):
                    new_offset = np.array(offset) + np.array(i_offsets[i])
                    new_key = (i_indice, tuple(new_offset))
                    
                    # more strict distance check
                    if (new_key not in visited_nodes and 
                        self._is_valid_distance(new_offset, atom_object, max_distance_factor=1.5)):
                        
                        cluster_nodes.append((i_indice, new_offset.tolist()))
                        visited_nodes.add(new_key)
                        queue.append((i_indice, new_offset.tolist(), depth + 1))
            except:
                continue
        
        return cluster_nodes
    
    def _build_enhanced_cluster(self, nl, cidx, atom_object):
        """
        Build an enhanced cluster, including all improvements
        
        Parameters
        ----------
        nl : NeighborList
            Neighbor list object
        cidx : int
            Cluster center index
        atom_object : ase.Atoms
            Atomic structure object
        
        Returns
        -------
        cluster_nodes : list[tuple]
            List of cluster nodes: [(node index, offset)]
        """
        cluster_nodes = []
        visited_nodes = set()
        visited_tag_nodes = set()
        
        # Phase 1: add the center node
        center_key = (cidx, (0, 0, 0))
        cluster_nodes.append((cidx, [0, 0, 0]))
        visited_nodes.add(center_key)
        
        # Phase 2: BFS search all nodes in the neighbor_depth range
        depth_limited_nodes = []
        queue = [(cidx, [0, 0, 0], 0)]
        
        while queue:
            node_idx, offset, depth = queue.pop(0)
            
            # reach the depth limit, stop the expansion of this branch
            if depth >= self.neighbor_depth:
                continue
            
            try:
                i_indices, i_offsets = nl.get_neighbors(node_idx)
                for i, i_indice in enumerate(i_indices):
                    new_offset = np.array(offset) + np.array(i_offsets[i])
                    new_key = (i_indice, tuple(new_offset))
                    
                    # avoid duplicate and check the distance threshold (prevent small lattice infinite loop)
                    if (new_key not in visited_nodes and 
                        self._is_valid_distance(new_offset, atom_object) and
                        self._should_include_image(i_indice, new_offset.tolist(), atom_object, visited_nodes)):
                        
                        cluster_nodes.append((i_indice, new_offset.tolist()))
                        depth_limited_nodes.append((i_indice, new_offset.tolist()))
                        visited_nodes.add(new_key)
                        queue.append((i_indice, new_offset.tolist(), depth + 1))
            except:
                continue
        
        # Phase 3: Search all connected tag nodes (no depth limit)
        # Collect all starting nodes for searching tag connectivity
        tag_search_nodes = []
        
        # 3.1 add the center node (if it is a tag node, or as a search start)
        tag_search_nodes.append((cidx, [0, 0, 0]))
        
        # 3.2 add the tag neighbors of each non-tag node in the depth range
        for node_idx, offset in depth_limited_nodes:
            if atom_object[node_idx].tag != self.tag:  # only process non-tag nodes
                tag_neighbors = self._find_tag_neighbors(nl, node_idx, offset, atom_object)
                tag_search_nodes.extend(tag_neighbors)
        
        # search all connected tag nodes of all starting nodes
        for start_node_idx, start_offset in tag_search_nodes:
            connected_tags = self._find_all_connected_tag_nodes(
                nl, start_node_idx, atom_object, visited_tag_nodes)
            
            for connected_tag_idx, connected_tag_offset in connected_tags:
                tag_key = (connected_tag_idx, tuple(connected_tag_offset))
                if tag_key not in visited_nodes:
                    cluster_nodes.append((connected_tag_idx, connected_tag_offset))
                    visited_nodes.add(tag_key)
        
        return cluster_nodes

    def _featurize(self, atom_object: Atoms, id: str, label: float, cidxs: list[int]) -> Data:
        """
        Calculate crystal graph features from pymatgen structure.

        Parameters
        ----------
        atom_object: ase.Atoms
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.
        id: str
            The unique identifier for the datapoint.
        label: float
            The target value for the datapoint.
        cidxs: list[int]
            The indexes of the cluster center in the structure.

        Returns
        -------
        data: torch_geometric.data.Data
            A cluster graph with CGCNN style features.
            
        Notes
        -----
        The with_pseudonodes parameter controls how nodes are filtered based on their connectivity:
        - with_pseudonodes=True: Exclude completely isolated nodes (indegree=0 AND outdegree=0), 
          keep all other nodes that have at least one incoming or outgoing edge
        - with_pseudonodes=False: Only keep nodes with incoming edges (indegree>0), 
          exclude nodes that are only source nodes or isolated nodes
        """

        node_features, edge_index, edge_features, node_weights = self._get_cluster_node_and_edge(atom_object, cidxs)
        data = GraphData(node_features, edge_index, edge_features, node_weights, id, label, cidxs).to_torch_geometric_data()
        return data
    
    def _get_cluster_node_and_edge(self, atom_object, cidxs):
        """
        Improved cluster node and edge acquisition method
        
        Parameters
        ----------
        atom_object: ase.Atoms
            The atomic structure.
        cidxs: list[int]
            The indexes of the cluster center in the structure.

        Returns
        -------
        node_features: np.ndarray
            A numpy array of shape `(num_nodes, 92)`.
        edge_index: np.ndarray, dtype int
            A numpy array of shape with `(2, num_edges)`.
        edge_features: np.ndarray
            A numpy array of shape with `(num_edges, filter_length)`.
        node_weights: np.ndarray
            A numpy array of shape with `(num_nodes, )`.
        """
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(atom_object)

        # get the index of all nodes
        crystal_nodes_idx = list(range(len(struct)))
        
        cluster_nodes_idx = []
        cluster_nodes_offsets = []
        
        if self.method == "CR":
            for cidx in cidxs:
                cidx_neighbors = struct.get_neighbors(struct[cidx], self.cluster_radius)
                # sort the neighbors by distance
                sorted_cidx_neighbors = sorted(cidx_neighbors, key=lambda x: x[1])
                cluster_nodes_idx.append(np.int64(cidx))
                cluster_nodes_offsets.append([0, 0, 0])
                cluster_nodes_idx.extend([x[2] for x in sorted_cidx_neighbors])
                cluster_nodes_offsets.extend([list(x[3]) for x in sorted_cidx_neighbors])

        elif self.method == "nth-NN":
            nl = NeighborList(
                natural_cutoffs(atom_object, self.neighbor_mult), 
                self_interaction=False, 
                bothways=True, 
                skin=0.25,
            )
            nl.update(atom_object)
            
            # detect small lattice structure
            is_small_lattice = self._detect_small_lattice(atom_object)
            
            for cidx in cidxs:
                if is_small_lattice and self.conservative_small_lattice:
                    # use conservative strategy for small lattice
                    cluster_nodes = self._build_conservative_cluster(nl, cidx, atom_object)
                else:
                    # use enhanced strategy for regular lattice
                    cluster_nodes = self._build_enhanced_cluster(nl, cidx, atom_object)
                
                # extract node index and offset
                for node_idx, offset in cluster_nodes:
                    cluster_nodes_idx.append(node_idx)
                    cluster_nodes_offsets.append(offset)
        
        # Limit cluster size if specified
        if self.max_cluster_nodes is not None and len(cluster_nodes_idx) > self.max_cluster_nodes:
            cluster_nodes_idx = cluster_nodes_idx[:self.max_cluster_nodes]
            cluster_nodes_offsets = cluster_nodes_offsets[:self.max_cluster_nodes]

        # find unique nodes index in cluster_nodes_idx
        unique_nodes_idx = list(set(cluster_nodes_idx))        
        node_weights = [cluster_nodes_idx.count(idx) for idx in crystal_nodes_idx]
        # get the irreducible cluster nodes
        nodes = [struct[idx] for idx in unique_nodes_idx]

        # get all neighbors for each node in the cluster
        neighbors = struct.get_all_neighbors(self.neighbor_radius, sites=nodes)
        neighbors = [sorted(n, key=lambda x: x[1]) for n in neighbors]
        neighbors = [n[: self.max_neighbors] for n in neighbors]

        src_idx, dest_idx = [], []
        edge_distances = []
        for node_idx, neighbor_list in zip(unique_nodes_idx, neighbors):
            for site in neighbor_list:
                src_idx.append(site[2])
                dest_idx.append(node_idx)
                edge_distances.append(site[1])

        # get edge features
        edge_features = self._gaussian_filter(np.array(edge_distances, dtype=np.float32))

        # get node features for all crystal nodes initially
        all_node_features = []
        for idx in crystal_nodes_idx:
            atomic_number = struct[idx].specie.number
            assert atomic_number in self.valid_atom_number
            all_node_features.append(self.atom_features[atomic_number])
        all_node_features = np.vstack(all_node_features).astype(np.float32)
        
        # Create edge index array
        edge_index = np.array([src_idx, dest_idx], dtype=np.int32)
        
        # Apply with_pseudonodes logic
        if edge_index.size > 0:
            # Calculate in-degree and out-degree for each node
            src_nodes = edge_index[0]  # source nodes (out-degree contributors)
            dest_nodes = edge_index[1]  # destination nodes (in-degree contributors)
            
            # Count degrees
            all_nodes = np.arange(len(crystal_nodes_idx))
            in_degrees = np.bincount(dest_nodes, minlength=len(crystal_nodes_idx))
            out_degrees = np.bincount(src_nodes, minlength=len(crystal_nodes_idx))
            
            if self.with_pseudonodes:
                # Keep nodes that are NOT completely isolated (indegree=0 AND outdegree=0)
                # This means keep nodes with indegree>0 OR outdegree>0
                nodes_to_keep = (in_degrees > 0) | (out_degrees > 0)
            else:
                # Only keep nodes with indegree > 0
                nodes_to_keep = in_degrees > 0
            
            # Get indices of nodes to keep
            kept_node_indices = np.where(nodes_to_keep)[0]
        else:
            # No edges case
            if self.with_pseudonodes:
                # Keep all nodes when no edges (since no nodes are isolated by edges)
                kept_node_indices = np.arange(len(crystal_nodes_idx))
            else:
                # Keep no nodes when no edges and only want nodes with indegree>0
                kept_node_indices = np.array([], dtype=np.int32)
        
        # Get edge feature dimension for consistent shape handling
        if edge_features.size > 0:
            edge_feature_dim = edge_features.shape[1]
        else:
            # Calculate default edge feature dimension when no edges exist
            # This is the dimension produced by _gaussian_filter
            filt = np.arange(0, self.neighbor_radius + self.step, self.step)
            edge_feature_dim = len(filt)
        
        # Create final outputs based on kept nodes
        if len(kept_node_indices) > 0:
            # Filter node features and weights for kept nodes only
            final_node_features = all_node_features[kept_node_indices]
            final_node_weights = np.array([node_weights[idx] for idx in kept_node_indices], dtype=np.float32)
            
            if edge_index.size > 0:
                # Create mapping from old indices to new indices
                old_to_new_mapping = {}
                for new_idx, old_idx in enumerate(kept_node_indices):
                    old_to_new_mapping[old_idx] = new_idx
                
                # Filter edges: only keep edges where both src and dest are in kept nodes
                valid_edges = []
                valid_edge_distances = []
                
                for i, (src, dest) in enumerate(zip(src_idx, dest_idx)):
                    if src in old_to_new_mapping and dest in old_to_new_mapping:
                        valid_edges.append([old_to_new_mapping[src], old_to_new_mapping[dest]])
                        valid_edge_distances.append(edge_distances[i])
                
                if valid_edges:
                    final_edge_index = np.array(valid_edges, dtype=np.int32).T
                    # Recalculate edge features for valid edges only
                    final_edge_features = self._gaussian_filter(np.array(valid_edge_distances, dtype=np.float32))
                else:
                    final_edge_index = np.array([[], []], dtype=np.int32)
                    final_edge_features = np.array([], dtype=np.float32).reshape(0, edge_feature_dim)
            else:
                final_edge_index = np.array([[], []], dtype=np.int32)
                final_edge_features = np.array([], dtype=np.float32).reshape(0, edge_feature_dim)
        else:
            # No nodes to keep
            final_node_features = np.empty((0, all_node_features.shape[1]), dtype=np.float32)
            final_node_weights = np.array([], dtype=np.float32)
            final_edge_index = np.array([[], []], dtype=np.int32)
            final_edge_features = np.array([], dtype=np.float32).reshape(0, edge_feature_dim)

        return final_node_features, final_edge_index, final_edge_features, final_node_weights

    def _gaussian_filter(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter to an array of interatomic distances.

        Parameters
        ----------
        distances : np.ndarray
            A numpy array of the shape `(num_edges, )`.

        Returns
        -------
        expanded_distances: np.ndarray
            Expanded distance tensor after Gaussian filtering.
            The shape is `(num_edges, filter_length)`. The `filter_length` is
            (self.neighbor_radius / self.step) + 1.
        """

        filt = np.arange(0, self.neighbor_radius + self.step, self.step)

        # Increase dimension of distance tensor and apply filter
        expanded_distances = np.exp(
            -((distances[..., np.newaxis] - filt) ** 2) / self.step**2
        )

        return expanded_distances 