from __future__ import annotations

import os
import json
import tqdm
import numpy as np
from typing import Tuple

try:
    import torch
    import dgl
    from dgl.data import DGLDataset
except ModuleNotFoundError:
    raise ImportError("This function requires DGL to be installed.")

from pymatgen.core.structure import Structure


class Featureizer:
    def __init__(
        self,
        radius: float = 8.0,
        max_neighbors: int = 12,
        cluster_radius: float = 10.0,
        max_nodes: int = 12,
        step: float = 0.2,
        json_path: str = "raw_dataset",
    ):
        """
        Parameters
        ----------
        radius: float, optional (default 8.0)
            The radius for searching neighbors.
        max_neighbors: int, optional (default 12)
            The maximum number of neighbors for each node considered.
        cluster_radius: float, optional (default 10.0)
            The radius for searching cluster atoms.
        max_nodes: int, optional (default 12)
            The maximum number of nodes in the cluster graph.
        step: float, optional (default 0.2)
            The step size for the Gaussian filter.
        json_path: str, optional (default ".")
            The path to the atom_init.json file.
        """
        self.radius = radius
        self.max_neighbors = max_neighbors
        self.cluster_radius = cluster_radius
        self.max_nodes = max_nodes
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

    def _featurize(self, datapoint: Structure, cidxs: list[int]) -> dgl.DGLGraph:
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
        graph = GraphData(node_features, node_weights, edge_index, edge_features).to_dgl_graph()
        return graph
    
    def _get_cluster_node_and_edge(self, struct: Structure, cidxs: list[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the node feature and edge feature from pymatgen structure.

        Parameters
        ----------
        struct: pymatgen.core.Structure
            A periodic crystal composed of a lattice and a sequence of atomic
            sites with 3D coordinates and elements.
        
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

        # get the index of all nodes
        crystal_nodes_idx = list(range(len(struct)))

        all_cluster_nodes = []
        all_cluster_nodes_idx = []
        for cidx in cidxs:
            # get cluster nodes
            cluster_nodes = struct.get_neighbors(struct[cidx], self.cluster_radius, include_index=True)
            cluster_nodes = sorted(cluster_nodes, key=lambda x: x[1])
            cluster_nodes = cluster_nodes[: self.max_nodes-1]
            # get the index of cluster nodes
            cluster_nodes_idx = [cidx]
            cluster_nodes_idx.extend([node[2] for node in cluster_nodes])
            all_cluster_nodes.extend(cluster_nodes)
            all_cluster_nodes_idx.extend(cluster_nodes_idx)
        
        # count the index of cluster nodes
        count = {}
        for idx in all_cluster_nodes_idx:
            count[idx] = count.get(idx, 0) + 1

        # remove duplicate index
        all_cluster_nodes_idx = list(set(all_cluster_nodes_idx))
        # get the irreducible cluster nodes
        all_cluster_nodes = [struct[idx] for idx in all_cluster_nodes_idx]

        # get all neighbors for each node in the cluster
        all_neighbors = struct.get_all_neighbors(self.radius, include_index=True, sites=all_cluster_nodes)
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
        from `data/sample-regression/atom_init.json` in the CGCNN repository.

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
            (self.radius / self.step) + 1. The edge features were built by applying gaussian
            filter to the distance between nodes.
        """

        neighbors = struct.get_all_neighbors(self.radius, include_index=True)
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
            (self.radius / self.step) + 1.
        """

        filt = np.arange(0, self.radius + self.step, self.step)

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
        non_isolated_mask = (in_degrees > 0) | (out_degrees > 0)
        non_isolated_nodes = torch.nonzero(non_isolated_mask, as_tuple=False).squeeze().int()
        g = g.subgraph(non_isolated_nodes)

        return g


class CGCNNDataset(DGLDataset):
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
    ):
        """
        Parameters
        ----------
        ids: list
            The unique identifier for each datapoint.
        cidxs: list[list[int]]
            The indexes of the cluster center in the structure.
        structures: list
            A list of pymatgen.core.Structure objects.
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
        """
        self.ids = ids
        self.cidxs = cidxs
        self.structures = structures
        self.labels = torch.tensor(labels, dtype=torch.float)
        self.featureizer = featureizer
        self.filename = filename
        self.save_cache = save_cache
        super(CGCNNDataset, self).__init__(name=name, save_dir=save_dir)

    def process(self):
        self.graphs = []
        num_graphs = len(self.structures)
        for i in tqdm.trange(num_graphs):
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
            torch.save(
                (self.ids, self.cidxs, self.graphs, self.labels),
                os.path.join(self.save_path, self.filename),
            )
            print(f"Saved {len(self.graphs)} datapoints to {self.save_path}")

    def load(self):
        self.ids, self.cidxs, self.graphs, self.labels = torch.load(
            os.path.join(self.save_path, self.filename)
        )
        print(f"Loaded {len(self.graphs)} datapoints from {self.save_path}")
        return self.ids, self.cidxs, self.graphs, self.labels
