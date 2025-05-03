from __future__ import annotations

import os
import json
import numpy as np
from typing import Tuple
from dask import delayed, compute
import dask
from tqdm import tqdm

try:
    import torch
    import dgl
    from dgl.data import DGLDataset
except ModuleNotFoundError:
    raise ImportError("This function requires DGL to be installed.")

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


class Featureizer:
    def __init__(
        self,
        neighbor_radius: float = 8.0,
        max_neighbors: int = 12,
        neighbor_depth: int = 2,
        neighbor_mult: float = 1.15,
        tag: int = 2,
        with_pseudonodes: bool = True,
        radius_search: bool = False,
        exclude_images: bool = False,
        cluster_radius: float = 3.0,
        max_cluster_nodes: int = 30,
        limit_cluster_nodes: bool = False,
        step: float = 0.2,
        json_path: str = "raw_dataset",
    ):
        """
        Parameters
        ----------
        neighbor_radius: float, optional (default 8.0)
            The radius for searching neighbors.
        max_neighbors: int, optional (default 12)
            The maximum number of neighbors for each node considered.
        neighbor_depth: int, optional (default 2)
            How many hops of neighbors to include for cluster node generation.
        neighbor_mult: float, optional (default 1.15)
            The multiplier for the natural cutoff radius.
        tag: int, optional (default 2)
            The tag of cluster atoms.
        with_pseudonodes: bool, optional (default True)
            Whether to include pseudonodes in the graph.
        radius_search: bool, optional (default False)
            Whether to perform a cluster search.
        exclude_images: bool, optional (default False)
            Whether to exclude images with tag in the cluster search.
        cluster_radius: float, optional (default 3.0)
            The radius for cluster search.
        max_cluster_nodes: int, optional (default 30)
            The maximum number of cluster nodes.
        step: float, optional (default 0.2)
            The step size for the Gaussian filter.
        json_path: str, optional (default ".")
            The path to the atom_init.json file.
        limit_cluster_nodes: bool, optional (default False)
            Whether to limit the number of cluster nodes.
        """
        self.neighbor_radius = neighbor_radius
        self.max_neighbors = max_neighbors
        self.neighbor_depth = neighbor_depth
        self.neighbor_mult = neighbor_mult
        self.tag = tag
        self.with_pseudonodes = with_pseudonodes
        self.radius_search = radius_search
        self.exclude_images = exclude_images
        self.cluster_radius = cluster_radius
        self.max_cluster_nodes = max_cluster_nodes
        self.limit_cluster_nodes = limit_cluster_nodes
        self.step = step
        self.json_path = json_path

        # load atom_init.json
        atom_init_json_path = os.path.join(self.json_path, "atom_init.json")
        with open(atom_init_json_path, "r") as f:
            atom_init_json = json.load(f)

        self.atom_features = {
            int(key): np.array(value, dtype=np.float32)
            for key, value in atom_init_json.items()
        }
        self.valid_atom_number = set(self.atom_features.keys())

    def _featurize(self, datapoint: Atoms, cidxs: list[int]) -> dgl.DGLGraph:
        """
        Calculate crystal graph features from pymatgen structure.

        Parameters
        ----------
        datapoint: pymatgen.core.Structure
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.

        cidxs: list[int]
            The indexes of the cluster center in the structure.

        Returns
        -------
        graph: dgl.DGLGraph
            A cluster graph with CGCNN style features.
        """

        node_features, node_weights, edge_index, edge_features = self._get_cluster_node_and_edge(datapoint, cidxs)
        graph = GraphData(node_features, node_weights, edge_index, edge_features, self.with_pseudonodes).to_dgl_graph()
        return graph
    
    def _get_cluster_node_and_edge(
        self, atom_object: Atoms, cidxs: list[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the node feature and edge feature from pymatgen structure.

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
        node_weights: np.ndarray
            A numpy array of shape `(num_nodes, 1)`.
        edge_index: np.ndarray, dtype int
            A numpy array of shape with `(2, num_edges)`.
        edge_features: np.ndarray
            A numpy array of shape with `(num_edges, filter_length)`.
        """

        # convert ase Atoms to pymatgen Structure
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(atom_object)

        # get the index of all nodes
        crystal_nodes_idx = list(range(len(struct)))

        nl = NeighborList(
            natural_cutoffs(atom_object, self.neighbor_mult), self_interaction=False, bothways=True, skin=0.25,
        )
        nl.update(atom_object)

        all_cluster_nodes_idx = []
        all_cluster_nodes_offsets = []
        for cidx in cidxs:
            cluster_nodes_idx = [np.int64(cidx)]
            cluster_nodes_offsets = [[0,0,0]]
            queue = [(np.int64(cidx), [0,0,0], 0)]  # (node_idx, offset, depth)
            while queue:
                node_idx, offset, depth = queue.pop(0)
                if depth >= self.neighbor_depth and atom_object[node_idx].tag != self.tag:
                    continue
                i_indices, i_offsets = nl.get_neighbors(node_idx)
                for i, i_indice in enumerate(i_indices):
                    new_offset = np.array(offset) + np.array(i_offsets[i])
                    exists = False
                    for idx, off in zip(cluster_nodes_idx, cluster_nodes_offsets):
                        if i_indice == idx:
                            if np.array_equal(new_offset, off):
                                exists = True
                                break
                            elif self.exclude_images and atom_object[i_indice].tag == self.tag:
                                exists = True
                                break
                    if not exists:
                        cluster_nodes_idx.append(i_indice)
                        cluster_nodes_offsets.append(new_offset)
                        queue.append((i_indice, new_offset, depth + 1))
            if self.radius_search:
                cidx_neighbors = struct.get_neighbors(struct[cidx], self.cluster_radius)
                for cidx_neighbor in cidx_neighbors:
                    i = cidx_neighbor[2]
                    image = list(cidx_neighbor[3])
                    exists = False
                    for idx, off in zip(cluster_nodes_idx, cluster_nodes_offsets):
                        if i == idx and np.array_equal(image, off):
                            exists = True
                            break
                    if not exists:
                        cluster_nodes_idx.append(i)
                        cluster_nodes_offsets.append(image)
            all_cluster_nodes_idx.extend(cluster_nodes_idx)
            all_cluster_nodes_offsets.extend(cluster_nodes_offsets)

        if self.limit_cluster_nodes and len(all_cluster_nodes_idx) > self.max_cluster_nodes:
            all_cluster_nodes_with_dist = []
            for idx, image in zip(all_cluster_nodes_idx, all_cluster_nodes_offsets):
                min_dist = np.inf
                for cidx in cidxs:
                    dist = struct.get_distance(idx, cidx, image)
                    if dist < min_dist:
                        min_dist = dist
                all_cluster_nodes_with_dist.append((idx, min_dist))
            all_cluster_nodes_with_dist.sort(key=lambda x: x[1])
            all_cluster_nodes_idx = [x[0] for x in all_cluster_nodes_with_dist[:self.max_cluster_nodes]]

        # count the index of cluster nodes
        count = {}
        for idx in all_cluster_nodes_idx:
            count[idx] = count.get(idx, 0) + 1

        # remove duplicate index
        all_cluster_nodes_idx = list(set(all_cluster_nodes_idx))
        # get the irreducible cluster nodes
        all_cluster_nodes = [struct[idx] for idx in all_cluster_nodes_idx]

        # get all neighbors for each node in the cluster
        all_neighbors = struct.get_all_neighbors(self.neighbor_radius, sites=all_cluster_nodes)
        all_neighbors = [sorted(n, key=lambda x: x[1]) for n in all_neighbors]
        all_neighbors = [n[: self.max_neighbors] for n in all_neighbors]

        src_idx, dest_idx = [], []
        edge_distances = []
        for node_idx, neighbor in zip(all_cluster_nodes_idx, all_neighbors):
            for site in neighbor:
                src_idx.append(site[2])
                dest_idx.append(node_idx)
                edge_distances.append(site[1])

        # get edge features
        edge_features = self._gaussian_filter(np.array(edge_distances, dtype=np.float32))

        # get node features and node weights
        node_features = []
        node_weights = []
        for idx in crystal_nodes_idx:
            atomic_number = struct[idx].specie.number
            assert atomic_number in self.valid_atom_number
            node_features.append(self.atom_features[atomic_number])
            if idx in count:
                node_weights.append(count[idx])
            else:
                node_weights.append(0)

        return np.vstack(node_features).astype(np.float32), np.vstack(node_weights).astype(np.float32), np.array([src_idx, dest_idx], dtype=np.int32), edge_features

    def _get_node_features(self, struct: Structure) -> np.ndarray:
        """
        Get the node feature from `atom_init.json`. The `atom_init.json` was collected
        from `data/sample-regression/atom_init.json` in the CGNET repository.

        Parameters
        ----------
        struct: pymatgen.core.Structure
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.

        Returns
        -------
        node_features: np.ndarray
            A numpy array of shape `(num_nodes, 92)`.
        """
        node_features = []
        for site in struct:
            # check whether the atom feature exists or not
            assert site.specie.number in self.valid_atom_number
            node_features.append(self.atom_features[site.specie.number])
        return np.vstack(node_features).astype(np.float32)

    def _get_edge_features_and_index(
        self, struct: Structure
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the edge feature and edge index from pymatgen structure.

        Parameters
        ----------
        struct: pymatgen.core.Structure
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.

        Returns
        -------
        edge_idx np.ndarray, dtype int
            A numpy array of shape with `(2, num_edges)`.
        edge_features: np.ndarray
            A numpy array of shape with `(num_edges, filter_length)`. The `filter_length` is
            (self.neighbor_radius / self.step) + 1. The edge features were built by applying gaussian
            filter to the distance between nodes.
        """

        neighbors = struct.get_all_neighbors(self.neighbor_radius, include_index=True)
        neighbors = [sorted(n, key=lambda x: x[1]) for n in neighbors]

        # construct bi-directed graph
        src_idx, dest_idx = [], []
        edge_distances = []
        for node_idx, neighbor in enumerate(neighbors):
            neighbor = neighbor[: self.max_neighbors]
            src_idx.extend([node_idx] * len(neighbor))
            dest_idx.extend([site[2] for site in neighbor])
            edge_distances.extend([site[1] for site in neighbor])

        edge_idx = np.array([src_idx, dest_idx], dtype=np.int32)
        edge_features = self._gaussian_filter(
            np.array(edge_distances, dtype=np.float32)
        )
        return edge_idx, edge_features

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


class GraphData:
    def __init__(
        self,
        node_features: np.ndarray,
        node_weights: np.ndarray,
        edge_index: np.ndarray,
        edge_features: np.ndarray | None = None,
        with_pseudonodes: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        node_features: np.ndarray
            Node feature matrix with shape [num_nodes, num_node_features]
        node_weights: np.ndarray
            Node weight matrix with shape [num_nodes, 1]
        edge_index: np.ndarray, dtype int
            Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
            Edge feature matrix with shape [num_edges, num_edge_features]
        with_pseudonodes: bool, optional (default True)
            Whether to include pseudonodes in the graph.
        kwargs: optional
            Additional attributes and their values
        """
        # validate params
        if isinstance(node_features, np.ndarray) is False:
            raise ValueError("node_features must be np.ndarray.")

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
        self.node_weights = node_weights
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.with_pseudonodes = with_pseudonodes
        self.kwargs = kwargs
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

        for key, value in self.kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        """Returns a string containing the printable representation of the object"""
        cls = self.__class__.__name__
        node_features_str = str(list(self.node_features.shape))
        node_weights_str = str(list(self.node_weights.shape))
        edge_index_str = str(list(self.edge_index.shape))
        if self.edge_features is not None:
            edge_features_str = str(list(self.edge_features.shape))
        else:
            edge_features_str = "None"

        out = "%s(node_features=%s, node_weights=%s, edge_index=%s, edge_features=%s" % (
            cls,
            node_features_str,
            node_weights_str,
            edge_index_str,
            edge_features_str,
        )
        # Adding shapes of kwargs
        for key, value in self.kwargs.items():
            if isinstance(value, np.ndarray):
                out += ", " + key + "=" + str(list(value.shape))
            elif isinstance(value, str):
                out += ", " + key + "=" + value
            elif isinstance(value, int) or isinstance(value, float):
                out += ", " + key + "=" + str(value)
        out += ")"
        return out

    def to_dgl_graph(self, self_loop: bool = False):
        """Convert to DGL graph data instance

        Returns
        -------
        dgl.DGLGraph
            Graph data for DGL
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges from nodes
            to themselves. Default to False.

        Note
        ----
        This method requires DGL to be installed.
        """

        src = self.edge_index[0]
        dst = self.edge_index[1]

        g = dgl.graph(
            (torch.from_numpy(src).int(), torch.from_numpy(dst).int()),
            num_nodes=self.num_nodes,
        )
        g.ndata["x"] = torch.from_numpy(self.node_features).float()
        g.ndata['w'] = torch.from_numpy(self.node_weights).float()

        if self.edge_features is not None:
            g.edata["edge_attr"] = torch.from_numpy(self.edge_features).float()

        if self_loop:
            # This assumes that the edge features for self loops are full-zero tensors
            # In the future we may want to support featurization for self loops
            g.add_edges(np.arange(self.num_nodes), np.arange(self.num_nodes))

        in_degrees = g.in_degrees()
        out_degrees = g.out_degrees()
        if self.with_pseudonodes:
            # Remove isolated nodes
            non_isolated_mask = (in_degrees > 0) | (out_degrees > 0)
        else:
            # Remove isolated nodes and pseudonodes
            non_isolated_mask = (in_degrees > 0) & (out_degrees > 0)
        non_isolated_nodes = torch.nonzero(non_isolated_mask, as_tuple=False).squeeze().int()
        g = g.subgraph(non_isolated_nodes)

        return g


class CGNETDataset(DGLDataset):
    def __init__(
        self,
        ids: list,
        cidxs: list[list[int]],
        structures: list,
        labels: list,
        featureizer: Featureizer,
        name: str = "Dataset",
        filename: str = "data.pt",
        save_cache: bool = False,
        clean_cache: bool = False,
        save_dir: str | None = None,
        use_parallel: bool = False,  # 新增参数
    ):
        """
        Parameters
        ----------
        ids: list
            The unique identifier for each datapoint.
        cidxs: list[list[int]]
            The indexes of the cluster center in the structure.
        structures: list
            A list of ase Atoms objects.
        labels: list
            A list of target values.
        featureizer: Featureizer
            The featureizer object to featurize the structure.
        name: str, optional (default "Dataset")
            The name of the dataset.
        filename: str, optional (default "data.pt")
            The name of the file to save the dataset.
        save_cache: bool, optional (default False)
            Whether to save the dataset to the disk.
        clean_cache: bool, optional (default False)
            Whether to clean the cache.
        save_dir: str, optional (default None)
            The directory to save the dataset.
        use_parallel: bool, optional (default True)
            Whether to use parallel processing for graph generation.
        """
        self.ids = ids
        self.cidxs = cidxs
        self.structures = structures
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.featureizer = featureizer
        self.filename = filename
        self.save_cache = save_cache
        self.use_parallel = use_parallel  # 保存并行化开关
        super(CGNETDataset, self).__init__(name=name, save_dir=save_dir)

    def process(self, chunk_size: int = 1000):
        """
        Process the dataset by generating graphs.

        Parameters
        ----------
        chunk_size: int, optional (default 1000)
            The number of structures to process in each chunk (used in parallel mode).
        """
        self.graphs = []
        num_graphs = len(self.structures)

        if self.use_parallel:
            @delayed
            def generate_graphs_chunk(start_idx, end_idx):
                chunk_graphs = []
                for i in range(start_idx, end_idx):
                    struct = self.structures[i]
                    cidxs = self.cidxs[i]
                    graph = self.featureizer._featurize(struct, cidxs=cidxs)
                    chunk_graphs.append(graph)
                return chunk_graphs

            tasks = []
            for start_idx in range(0, num_graphs, chunk_size):
                end_idx = min(start_idx + chunk_size, num_graphs)
                tasks.append(generate_graphs_chunk(start_idx, end_idx))

            tasks_with_progress = tqdm(tasks, desc="Processing chunks", total=len(tasks))
            chunk_results = compute(*tasks_with_progress, scheduler="processes")
            self.graphs = [graph for chunk in chunk_results for graph in chunk]
        else:
            for i in tqdm(range(num_graphs), desc="Processing graphs"):
                struct = self.structures[i]
                cidxs = self.cidxs[i]
                graph = self.featureizer._featurize(struct, cidxs=cidxs)
                self.graphs.append(graph)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.ids[idx], self.cidxs[idx], self.graphs[idx], self.labels[idx]

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.filename))

    def save(self):
        if self.save_cache:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            save_filename = os.path.join(self.save_path, self.filename)
            torch.save(
                (self.ids, self.cidxs, self.graphs, self.labels),
                save_filename,
            )
            print(f"Saved {len(self.graphs)} datapoints to {save_filename}")

    def load(self):
        load_filename = os.path.join(self.save_path, self.filename)
        self.ids, self.cidxs, self.graphs, self.labels = torch.load(
            load_filename
        )
        print(f"Loaded {len(self.graphs)} datapoints from {load_filename}")
        return self.ids, self.cidxs, self.graphs, self.labels
