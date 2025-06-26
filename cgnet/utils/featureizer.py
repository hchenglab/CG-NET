from __future__ import annotations

import os
import json
import numpy as np
from typing import Tuple, List, Optional, Dict, Set

try:
    from torch_geometric.data import Data
except ModuleNotFoundError:
    raise ImportError("This function requires PyTorch Geometric to be installed.")

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs

from .data import GraphData


class CGNETFeatureizer:
    """
    A featurizer for converting crystal structures to graph representations.
    
    Supports two methods:
    - CR (Cluster Radius): Fixed radius clustering
    - nth-NN (nth Nearest Neighbor): Neighbor depth-based clustering
    """
    
    def __init__(self, 
                 method: str = "CR",
                 neighbor_radius: float = 8.0,
                 max_neighbors: int = 12,
                 # CR method parameters
                 cluster_radius: float = 7.0,
                 # nth-NN method parameters
                 neighbor_depth: int = 2,
                 neighbor_mult: float = 1.15,
                 max_distance_factor: float = 2.0,
                 small_lattice_threshold: float = 5.0,
                 enable_smart_images: bool = True,
                 conservative_small_lattice: bool = False,
                 # Common parameters
                 max_cluster_nodes: Optional[int] = None,
                 tag: int = 2,
                 step: float = 0.2,
                 with_pseudonodes: bool = True,
                 json_path: Optional[str] = None,
                 json_file: str = "atom_init.json",
                 **kwargs):
        """
        Initialize the CGNETFeatureizer.

        Parameters
        ----------
        method : str, default "CR"
            Featurization method. Options: "CR", "nth-NN"
        neighbor_radius : float, default 8.0
            Cutoff radius for neighbor search
        max_neighbors : int, default 12
            Maximum number of neighbors per atom
        cluster_radius : float, default 7.0
            Cluster radius for CR method
        neighbor_depth : int, default 2
            Neighbor search depth for nth-NN method
        neighbor_mult : float, default 1.15
            Neighbor radius multiplier for nth-NN method
        max_distance_factor : float, default 2.0
            Maximum distance factor for nth-NN method
        small_lattice_threshold : float, default 5.0
            Threshold for small lattice detection
        enable_smart_images : bool, default True
            Enable smart image selection
        conservative_small_lattice : bool, default False
            Use conservative strategy for small lattices
        max_cluster_nodes : int or None, default None
            Maximum number of cluster nodes
        tag : int, default 2
            Tag for cluster centers
        step : float, default 0.2
            Step size for Gaussian filter
        with_pseudonodes : bool, default True
            Include pseudonodes in graph
        json_path : str or None, default None
            Path to atom features JSON file
        json_file : str, default "atom_init.json"
            Name of atom features JSON file
        """
        self._validate_and_set_parameters(
            method, neighbor_radius, max_neighbors, cluster_radius,
            neighbor_depth, neighbor_mult, max_distance_factor,
            small_lattice_threshold, enable_smart_images, conservative_small_lattice,
            max_cluster_nodes, tag, step, with_pseudonodes, json_path, json_file
        )
        self._load_atom_features()
        
    def _validate_and_set_parameters(self, method: str, neighbor_radius: float, 
                                   max_neighbors: int, cluster_radius: float,
                                   neighbor_depth: int, neighbor_mult: float,
                                   max_distance_factor: float, small_lattice_threshold: float,
                                   enable_smart_images: bool, conservative_small_lattice: bool,
                                   max_cluster_nodes: Optional[int], tag: int, step: float,
                                   with_pseudonodes: bool, json_path: Optional[str], json_file: str):
        """Validate and set all parameters."""
        if method not in ["CR", "nth-NN"]:
            raise ValueError(f"Method must be 'CR' or 'nth-NN', got '{method}'")
            
        self.method = method
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.max_cluster_nodes = max_cluster_nodes
        self.tag = tag
        self.step = step
        self.with_pseudonodes = with_pseudonodes
        
        # Set JSON path
        self.json_path = json_path or os.path.dirname(os.path.abspath(__file__))
        self.json_file = json_file
        
        # Method-specific parameters
        if method == "CR":
            if cluster_radius <= 0:
                raise ValueError("cluster_radius must be positive for CR method")
            self.cluster_radius = cluster_radius
            print(f"Using CR method with cluster radius {cluster_radius}, max nodes {max_cluster_nodes}")
            
        elif method == "nth-NN":
            if neighbor_depth <= 0:
                raise ValueError("neighbor_depth must be positive for nth-NN method")
            self.neighbor_depth = neighbor_depth
            self.neighbor_mult = neighbor_mult
            self.max_distance_factor = max_distance_factor
            self.small_lattice_threshold = small_lattice_threshold
            self.enable_smart_images = enable_smart_images
            self.conservative_small_lattice = conservative_small_lattice
            print(f"Using nth-NN method with depth {neighbor_depth}, max nodes {max_cluster_nodes}")
    
    def _load_atom_features(self) -> None:
        """Load atom features from JSON file."""
        atom_features_path = os.path.join(self.json_path, self.json_file)
        
        if not os.path.exists(atom_features_path):
            raise FileNotFoundError(f"Atom features file not found: {atom_features_path}")
            
        try:
            with open(atom_features_path, "r") as f:
                atom_init_json = json.load(f)
                
            self.atom_features = {
                int(key): np.array(value, dtype=np.float32)
                for key, value in atom_init_json.items()
            }
            self.valid_atom_number = set(self.atom_features.keys())
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid atom features file format: {e}")
    
    def _detect_small_lattice(self, atom_object: Atoms, threshold: Optional[float] = None) -> bool:
        """
        Detect if structure has small lattice parameters.
        
        Parameters
        ----------
        atom_object : Atoms
            Crystal structure
        threshold : float, optional
            Small lattice threshold in Angstroms
            
        Returns
        -------
        bool
            True if structure has small lattice parameters
        """
        threshold = threshold or self.small_lattice_threshold
        
        cell = atom_object.get_cell()
        cell_lengths = np.linalg.norm(cell, axis=1)
        valid_lengths = cell_lengths[cell_lengths > 1e-6]
        
        return len(valid_lengths) > 0 and np.any(valid_lengths < threshold)
    
    def _is_valid_distance(self, offset: np.ndarray, atom_object: Atoms, 
                          max_distance_factor: Optional[float] = None) -> bool:
        """
        Check if periodic offset represents valid distance.
        
        Parameters
        ----------
        offset : np.ndarray
            Periodic offset vector
        atom_object : Atoms
            Crystal structure
        max_distance_factor : float, optional
            Maximum distance factor relative to cell size
            
        Returns
        -------
        bool
            True if distance is valid
        """
        offset = np.asarray(offset)
        
        # Zero offset is always valid
        if np.allclose(offset, 0):
            return True
            
        max_distance_factor = max_distance_factor or self.max_distance_factor
        
        # Calculate real space distance
        cell = atom_object.get_cell()
        real_offset = np.dot(offset, cell)
        distance = np.linalg.norm(real_offset)
        
        # Determine distance threshold
        cell_lengths = np.linalg.norm(cell, axis=1)
        valid_lengths = cell_lengths[cell_lengths > 1e-6]
        
        if len(valid_lengths) == 0:
            # Molecular system
            distance_threshold = self.neighbor_radius * 2.0
        else:
            min_cell_length = np.min(valid_lengths)
            distance_threshold = max(
                min_cell_length * max_distance_factor,
                self.neighbor_radius,
                10.0  # Absolute minimum
            )
        
        return distance < distance_threshold
    
    def _get_node_neighbors(self, nl: NeighborList, node_idx: int, 
                           atom_object: Atoms) -> Tuple[np.ndarray, np.ndarray]:
        """Get neighbors for a node, handling exceptions gracefully."""
        try:
            return nl.get_neighbors(node_idx)
        except Exception:
            # Return empty arrays if neighbor lookup fails
            return np.array([], dtype=int), np.array([]).reshape(0, 3)
    
    def _find_all_connected_tag_nodes(self, nl: NeighborList, start_idx: int, 
                                     atom_object: Atoms, 
                                     visited_tag_nodes: Optional[Set] = None) -> List[Tuple[int, List[int]]]:
        """
        Find all tag nodes connected to starting node via BFS.
        
        Parameters
        ----------
        nl : NeighborList
            Neighbor list object
        start_idx : int
            Starting node index
        atom_object : Atoms
            Crystal structure
        visited_tag_nodes : set, optional
            Previously visited tag nodes
            
        Returns
        -------
        List[Tuple[int, List[int]]]
            Connected tag nodes as (index, offset) pairs
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
            
            # Add tag node if not visited
            if (atom_object[node_idx].tag == self.tag and 
                node_key not in visited_tag_nodes):
                tag_nodes.append((node_idx, offset))
                visited_tag_nodes.add(node_key)
                
                # Search neighbors of tag node
                i_indices, i_offsets = self._get_node_neighbors(nl, node_idx, atom_object)
                
                for i, i_indice in enumerate(i_indices):
                    new_offset = np.array(offset) + np.array(i_offsets[i])
                    new_key = (i_indice, tuple(new_offset))
                    
                    if (atom_object[i_indice].tag == self.tag and 
                        new_key not in visited_tag_nodes and 
                        new_key not in local_visited and
                        self._is_valid_distance(new_offset, atom_object)):
                        queue.append((i_indice, new_offset.tolist()))
        
        return tag_nodes
    
    def _find_tag_neighbors(self, nl: NeighborList, node_idx: int, offset: List[int], 
                           atom_object: Atoms) -> List[Tuple[int, List[int]]]:
        """Find tag neighbors of specified node."""
        tag_neighbors = []
        i_indices, i_offsets = self._get_node_neighbors(nl, node_idx, atom_object)
        
        for i, i_indice in enumerate(i_indices):
            if atom_object[i_indice].tag == self.tag:
                new_offset = np.array(offset) + np.array(i_offsets[i])
                if self._is_valid_distance(new_offset, atom_object):
                    tag_neighbors.append((i_indice, new_offset.tolist()))
        
        return tag_neighbors
    
    def _should_include_image(self, node_idx: int, offset: List[int], 
                             atom_object: Atoms, visited_nodes: Set) -> bool:
        """Determine if image atom should be included."""
        if not self.enable_smart_images:
            return True
        
        node_key = (node_idx, tuple(offset))
        
        if node_key in visited_nodes:
            return False
        
        if not self._is_valid_distance(np.array(offset), atom_object):
            return False
        
        # Relaxed strategy for tag nodes
        if atom_object[node_idx].tag == self.tag:
            return True
        
        # Distance check for non-tag nodes
        if not np.allclose(offset, 0):
            cell = atom_object.get_cell()
            real_offset = np.dot(offset, cell)
            distance = np.linalg.norm(real_offset)
            return distance <= self.neighbor_radius * 1.5
        
        return True
    
    def _build_conservative_cluster(self, nl: NeighborList, cidx: int, 
                                   atom_object: Atoms) -> List[Tuple[int, List[int]]]:
        """Build conservative cluster for small lattices."""
        cluster_nodes = [(cidx, [0, 0, 0])]
        visited_nodes = {(cidx, (0, 0, 0))}
        
        queue = [(cidx, [0, 0, 0], 0)]
        max_depth = min(self.neighbor_depth, 2)
        
        while queue:
            node_idx, offset, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            i_indices, i_offsets = self._get_node_neighbors(nl, node_idx, atom_object)
            
            for i, i_indice in enumerate(i_indices):
                new_offset = np.array(offset) + np.array(i_offsets[i])
                new_key = (i_indice, tuple(new_offset))
                
                if (new_key not in visited_nodes and 
                    self._is_valid_distance(new_offset, atom_object, max_distance_factor=1.5)):
                    
                    cluster_nodes.append((i_indice, new_offset.tolist()))
                    visited_nodes.add(new_key)
                    queue.append((i_indice, new_offset.tolist(), depth + 1))
        
        return cluster_nodes
    
    def _build_enhanced_cluster(self, nl: NeighborList, cidx: int, 
                               atom_object: Atoms) -> List[Tuple[int, List[int]]]:
        """Build enhanced cluster with tag node connectivity."""
        cluster_nodes = [(cidx, [0, 0, 0])]
        visited_nodes = {(cidx, (0, 0, 0))}
        visited_tag_nodes = set()
        
        # Phase 1: BFS within neighbor depth
        depth_limited_nodes = []
        queue = [(cidx, [0, 0, 0], 0)]
        
        while queue:
            node_idx, offset, depth = queue.pop(0)
            
            if depth >= self.neighbor_depth:
                continue
            
            i_indices, i_offsets = self._get_node_neighbors(nl, node_idx, atom_object)
            
            for i, i_indice in enumerate(i_indices):
                new_offset = np.array(offset) + np.array(i_offsets[i])
                new_key = (i_indice, tuple(new_offset))
                
                if (new_key not in visited_nodes and 
                    self._is_valid_distance(new_offset, atom_object) and
                    self._should_include_image(i_indice, new_offset.tolist(), atom_object, visited_nodes)):
                    
                    cluster_nodes.append((i_indice, new_offset.tolist()))
                    depth_limited_nodes.append((i_indice, new_offset.tolist()))
                    visited_nodes.add(new_key)
                    queue.append((i_indice, new_offset.tolist(), depth + 1))
        
        # Phase 2: Find connected tag nodes
        tag_search_nodes = [(cidx, [0, 0, 0])]
        
        # Add tag neighbors of non-tag nodes
        for node_idx, offset in depth_limited_nodes:
            if atom_object[node_idx].tag != self.tag:
                tag_neighbors = self._find_tag_neighbors(nl, node_idx, offset, atom_object)
                tag_search_nodes.extend(tag_neighbors)
        
        # Search all connected tag nodes
        for start_node_idx, start_offset in tag_search_nodes:
            connected_tags = self._find_all_connected_tag_nodes(
                nl, start_node_idx, atom_object, visited_tag_nodes)
            
            for connected_tag_idx, connected_tag_offset in connected_tags:
                tag_key = (connected_tag_idx, tuple(connected_tag_offset))
                if tag_key not in visited_nodes:
                    cluster_nodes.append((connected_tag_idx, connected_tag_offset))
                    visited_nodes.add(tag_key)
        
        return cluster_nodes

    def _featurize(self, atom_object: Atoms, id: str, label: float, cidxs: List[int]) -> Data:
        """
        Convert crystal structure to graph representation.

        Parameters
        ----------
        atom_object : Atoms
            Crystal structure
        id : str
            Unique identifier for datapoint
        label : float
            Target value for datapoint
        cidxs : List[int]
            Indices of cluster centers

        Returns
        -------
        Data
            PyTorch Geometric data object with graph features
        """
        node_features, edge_index, edge_features, node_weights = (
            self._get_cluster_node_and_edge(atom_object, cidxs)
        )
        
        data = GraphData(
            node_features, edge_index, edge_features, 
            node_weights, id, label, cidxs
        ).to_torch_geometric_data()
        
        return data
    
    def _get_cluster_node_and_edge(self, atom_object: Atoms, 
                                  cidxs: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract cluster nodes and edges from crystal structure.

        Parameters
        ----------
        atom_object : Atoms
            Crystal structure
        cidxs : List[int]
            Cluster center indices

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Node features, edge indices, edge features, node weights
        """
        # Get structure and basic setup
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(atom_object)
        crystal_nodes_idx = list(range(len(struct)))
        
        # Get cluster nodes based on method
        cluster_nodes_idx, cluster_nodes_offsets = self._get_cluster_nodes(
            struct, atom_object, cidxs
        )
        
        # Apply cluster size limit
        if (self.max_cluster_nodes is not None and 
            len(cluster_nodes_idx) > self.max_cluster_nodes):
            cluster_nodes_idx = cluster_nodes_idx[:self.max_cluster_nodes]
            cluster_nodes_offsets = cluster_nodes_offsets[:self.max_cluster_nodes]

        # Calculate node weights and get unique nodes
        unique_nodes_idx = list(set(cluster_nodes_idx))
        node_weights = [cluster_nodes_idx.count(idx) for idx in crystal_nodes_idx]
        
        # Get neighbors and edges
        edge_data = self._get_edge_data(struct, unique_nodes_idx)
        
        # Get all node features
        all_node_features = self._get_all_node_features(struct, crystal_nodes_idx)
        
        # Apply node filtering and create final outputs
        return self._apply_node_filtering(
            all_node_features, node_weights, edge_data, crystal_nodes_idx
        )
    
    def _get_cluster_nodes(self, struct: Structure, atom_object: Atoms, 
                          cidxs: List[int]) -> Tuple[List[int], List[List[int]]]:
        """Get cluster nodes based on selected method."""
        cluster_nodes_idx = []
        cluster_nodes_offsets = []
        
        if self.method == "CR":
            for cidx in cidxs:
                cidx_neighbors = struct.get_neighbors(struct[cidx], self.cluster_radius)
                sorted_neighbors = sorted(cidx_neighbors, key=lambda x: x[1])
                
                cluster_nodes_idx.append(np.int64(cidx))
                cluster_nodes_offsets.append([0, 0, 0])
                cluster_nodes_idx.extend([x[2] for x in sorted_neighbors])
                cluster_nodes_offsets.extend([list(x[3]) for x in sorted_neighbors])
                
        elif self.method == "nth-NN":
            nl = NeighborList(
                natural_cutoffs(atom_object, self.neighbor_mult),
                self_interaction=False,
                bothways=True,
                skin=0.25,
            )
            nl.update(atom_object)
            
            is_small_lattice = self._detect_small_lattice(atom_object)
            
            for cidx in cidxs:
                if is_small_lattice and self.conservative_small_lattice:
                    cluster_nodes = self._build_conservative_cluster(nl, cidx, atom_object)
                else:
                    cluster_nodes = self._build_enhanced_cluster(nl, cidx, atom_object)
                
                for node_idx, offset in cluster_nodes:
                    cluster_nodes_idx.append(node_idx)
                    cluster_nodes_offsets.append(offset)
        
        return cluster_nodes_idx, cluster_nodes_offsets
    
    def _get_edge_data(self, struct: Structure, 
                      unique_nodes_idx: List[int]) -> Dict[str, List]:
        """Get edge indices and distances."""
        nodes = [struct[idx] for idx in unique_nodes_idx]
        neighbors = struct.get_all_neighbors(self.neighbor_radius, sites=nodes)
        neighbors = [sorted(n, key=lambda x: x[1])[:self.max_neighbors] for n in neighbors]
        
        src_idx, dest_idx, edge_distances = [], [], []
        
        for node_idx, neighbor_list in zip(unique_nodes_idx, neighbors):
            for site in neighbor_list:
                src_idx.append(site[2])
                dest_idx.append(node_idx)
                edge_distances.append(site[1])
        
        return {
            'src_idx': src_idx,
            'dest_idx': dest_idx,
            'edge_distances': edge_distances
        }
    
    def _get_all_node_features(self, struct: Structure, 
                              crystal_nodes_idx: List[int]) -> np.ndarray:
        """Get node features for all crystal nodes."""
        all_node_features = []
        
        for idx in crystal_nodes_idx:
            atomic_number = struct[idx].specie.number
            if atomic_number not in self.valid_atom_number:
                raise ValueError(
                    f"Atomic number {atomic_number} not supported. "
                    f"Supported: {sorted(self.valid_atom_number)}"
                )
            all_node_features.append(self.atom_features[atomic_number])
        
        return np.vstack(all_node_features).astype(np.float32)
    
    def _apply_node_filtering(self, all_node_features: np.ndarray, node_weights: List[int],
                             edge_data: Dict[str, List], 
                             crystal_nodes_idx: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Apply pseudonode filtering logic."""
        src_idx = edge_data['src_idx']
        dest_idx = edge_data['dest_idx']
        edge_distances = edge_data['edge_distances']
        
        # Create edge index array
        edge_index = np.array([src_idx, dest_idx], dtype=np.int32) if src_idx else np.array([[], []], dtype=np.int32)
        
        # Calculate edge feature dimension
        filt = np.arange(0, self.neighbor_radius + self.step, self.step)
        edge_feature_dim = len(filt)
        
        # Determine nodes to keep based on connectivity
        if edge_index.size > 0:
            in_degrees = np.bincount(edge_index[1], minlength=len(crystal_nodes_idx))
            out_degrees = np.bincount(edge_index[0], minlength=len(crystal_nodes_idx))
            
            if self.with_pseudonodes:
                # Keep nodes with any connectivity
                nodes_to_keep = (in_degrees > 0) | (out_degrees > 0)
            else:
                # Only keep nodes with incoming edges
                nodes_to_keep = in_degrees > 0
                
            kept_node_indices = np.where(nodes_to_keep)[0]
        else:
            # No edges case
            if self.with_pseudonodes:
                kept_node_indices = np.arange(len(crystal_nodes_idx))
            else:
                kept_node_indices = np.array([], dtype=np.int32)
        
        # Create final outputs
        if len(kept_node_indices) > 0:
            final_node_features = all_node_features[kept_node_indices]
            final_node_weights = np.array([node_weights[idx] for idx in kept_node_indices], dtype=np.float32)
            
            if edge_index.size > 0:
                final_edge_index, final_edge_features = self._filter_edges(
                    edge_index, edge_distances, kept_node_indices, edge_feature_dim
                )
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
    
    def _filter_edges(self, edge_index: np.ndarray, edge_distances: List[float],
                     kept_node_indices: np.ndarray, 
                     edge_feature_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Filter edges based on kept nodes."""
        # Create mapping from old to new indices
        old_to_new_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_node_indices)}
        
        # Filter edges
        valid_edges = []
        valid_edge_distances = []
        
        for i, (src, dest) in enumerate(edge_index.T):
            if src in old_to_new_mapping and dest in old_to_new_mapping:
                valid_edges.append([old_to_new_mapping[src], old_to_new_mapping[dest]])
                valid_edge_distances.append(edge_distances[i])
        
        if valid_edges:
            final_edge_index = np.array(valid_edges, dtype=np.int32).T
            final_edge_features = self._gaussian_filter(np.array(valid_edge_distances, dtype=np.float32))
        else:
            final_edge_index = np.array([[], []], dtype=np.int32)
            final_edge_features = np.array([], dtype=np.float32).reshape(0, edge_feature_dim)
        
        return final_edge_index, final_edge_features

    def _gaussian_filter(self, distances: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter to interatomic distances.

        Parameters
        ----------
        distances : np.ndarray
            Array of distances with shape (num_edges,)

        Returns
        -------
        np.ndarray
            Gaussian-filtered distances with shape (num_edges, filter_length)
        """
        if distances.size == 0:
            filt = np.arange(0, self.neighbor_radius + self.step, self.step)
            return np.empty((0, len(filt)), dtype=np.float32)
        
        filt = np.arange(0, self.neighbor_radius + self.step, self.step)
        expanded_distances = np.exp(
            -((distances[..., np.newaxis] - filt) ** 2) / self.step**2
        )
        
        return expanded_distances.astype(np.float32) 