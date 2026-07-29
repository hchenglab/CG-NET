import json
import os

import torch
import torch.nn as nn
from fairchem.core.common.registry import registry
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.base import HeadInterface
from fairchem.core.graph.compute import generate_graph


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(-1).expand_as(src)
    return out.scatter_add_(0, index_expanded, src)

class CGCNNConv(nn.Module):
    """
    Convolutional operation on graphs for CGCNN natively.
    """
    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super(CGCNNConv, self).__init__() # CGCNN sums neighbors
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * self.atom_fea_len + self.nbr_fea_len,
                                 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor):
        # x: [N, atom_fea_len]
        # edge_index: [2, E]
        # edge_attr: [E, nbr_fea_len]
        
        # message passing
        row, col = edge_index
        x_i = x[row]
        x_j = x[col]
        
        msg = self.message(x_i, x_j, edge_attr)
        out = scatter_sum(msg, col, dim_size=x.size(0))
        
        # Check if batch size is 1 to avoid batchnorm error on size 1
        if out.shape[0] > 1:
            out = self.bn2(out)
            
        out = self.softplus2(x + out)
        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        # x_i: [E, atom_fea_len] (source node / neighbor)
        # x_j: [E, atom_fea_len] (target node / center atom)
        # Original CGCNN concatenates: [center_atom, neighbor_atom, edge_attr] -> [x_j, x_i, edge_attr]
        total_nbr_fea = torch.cat([x_j, x_i, edge_attr], dim=1)
        total_gated_fea = self.fc_full(total_nbr_fea)
        if total_gated_fea.shape[0] > 1:
            total_gated_fea = self.bn1(total_gated_fea)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=1)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        
        return nbr_filter * nbr_core


@registry.register_model("cgcnn_backbone")
class CGCNNBackbone(nn.Module):
    def __init__(self, 
                 orig_atom_fea_len: int = 92, 
                 nbr_fea_len: int = 41, 
                 atom_fea_len: int = 64, 
                 n_conv: int = 3, 
                 cutoff: float = 8.0,
                 dmin: float = 0.0,
                 otf_graph: bool = False,
                 max_neighbors: int = 30,
                 **kwargs):
        super().__init__()
        self.orig_atom_fea_len = orig_atom_fea_len
        self.atom_fea_len = atom_fea_len
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        
        # 1. Load atom_init.json to initialize fixed embedding
        json_path = os.path.join(os.path.dirname(__file__), "atom_init.json")
        with open(json_path) as f:
            atom_init = json.load(f)
        
        max_z = max(int(k) for k in atom_init.keys())
        embed_weights = torch.zeros((max_z + 1, orig_atom_fea_len))
        for z_str, feats in atom_init.items():
            embed_weights[int(z_str)] = torch.tensor(feats, dtype=torch.float32)
            
        self.atom_embedding = nn.Embedding.from_pretrained(embed_weights, freeze=True)
        
        # 2. Gaussian Smearing for edge distances (Original CGCNN logic)
        self.dmin = dmin
        self.cutoff = cutoff
        self.step = (self.cutoff - self.dmin) / (nbr_fea_len - 1) if nbr_fea_len > 1 else 0.2
        self.filter = nn.Parameter(torch.arange(self.dmin, self.cutoff + self.step * 0.5, self.step), requires_grad=False)
        self.var = self.step
        
        # 3. Model components
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([CGCNNConv(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])


    def _get_edge_attr(self, distances: torch.Tensor):
        # CGCNN's Gaussian Distance
        return torch.exp(-(distances - self.filter.view(1, -1))**2 / self.var**2)

    def _generate_graph(self, data: AtomicData):
        if self.otf_graph:
            pbc = data.get("pbc", torch.zeros(len(data), 3, dtype=torch.bool, device=data.pos.device))
            cluster_partition = data.get("cluster_partition", None)
            
            graph_dict = generate_graph(
                data,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                enforce_max_neighbors_strictly=False,
                radius_pbc_version=2,
                pbc=pbc,
                cluster_partition=cluster_partition,
            )
            # calculate distance from vector
            graph_dict["edge_distance"] = torch.norm(graph_dict["edge_distance_vec"], dim=-1)
        else:
            edge_index = data.edge_index
            row, col = edge_index
            if hasattr(data, "edge_dist") and data.edge_dist is not None:
                edge_dist = data.edge_dist
            else:
                if hasattr(data, "cell_offsets"):
                    out = data.pos[col] - data.pos[row] + torch.einsum('ni,nij->nj', data.cell_offsets, data.cell)
                else:
                    out = data.pos[col] - data.pos[row]
                edge_dist = torch.norm(out, dim=-1)
            
            graph_dict = {
                "edge_index": edge_index,
                "edge_distance": edge_dist
            }
        return graph_dict

    def forward(self, data: AtomicData) -> dict[str, torch.Tensor]:
        # Atomic features
        z = data.atomic_numbers.long()
        x = self.atom_embedding(z) # [N, 92]
        x = self.embedding(x) # [N, atom_fea_len]
        
        # Graph generation
        graph_dict = self._generate_graph(data)
        edge_index = graph_dict["edge_index"]
        edge_dist = graph_dict["edge_distance"]
        
        if edge_dist.dim() == 1:
            edge_dist = edge_dist.unsqueeze(-1)
            
        edge_attr = self._get_edge_attr(edge_dist) # [E, nbr_fea_len]
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            
        return {"node_features": x}


@registry.register_model("cgcnn_system_scalar_head")
class CGCNN_System_Scalar_Head(nn.Module, HeadInterface):
    """
    Head for system level scalars using CGCNN's MLP logic.
    Applies graph pooling FIRST, then nonlinear layers.
    Includes atom_weight support for pseudo-nodes.
    """
    def __init__(self, 
                 backbone: CGCNNBackbone, 
                 output_property: str = "energy", 
                 h_fea_len: int = 128, 
                 n_h: int = 1,
                 **kwargs):
        super().__init__()
        self.output_property = output_property
        self.conv_to_fc = nn.Linear(backbone.atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.n_h = n_h
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
            
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atom_fea = emb["node_features"]
        batch = data_dict.batch
        
        # 1. Weighted Global Mean Pooling
        atom_weight = data_dict.get("atom_weight", None)
        num_graphs = batch.max().item() + 1 if batch.numel() > 0 else 0
        if atom_weight is not None:
            # Weighted mean pool to discard pseudo-nodes (where weight = 0)
            summed = scatter_sum(atom_fea * atom_weight.unsqueeze(-1), batch, dim_size=num_graphs)
            weights_sum = torch.zeros(num_graphs, dtype=atom_weight.dtype, device=atom_weight.device)
            weights_sum.scatter_add_(0, batch, atom_weight)
            weights_sum = weights_sum.unsqueeze(-1)
            crys_fea = summed / weights_sum.clamp(min=1e-6)
        else:
            # Standard mean pool
            summed = scatter_sum(atom_fea, batch, dim_size=num_graphs)
            counts = torch.zeros(num_graphs, dtype=atom_fea.dtype, device=atom_fea.device)
            counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=atom_fea.dtype))
            crys_fea = summed / counts.clamp(min=1).unsqueeze(-1)
            
        # 2. Fully connected layers
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.n_h > 1:
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
                
        out = self.fc_out(crys_fea)
        return {self.output_property: out.squeeze(-1) if out.shape[-1] == 1 else out}


@registry.register_model("cgcnn_atom_scalar_head")
class CGCNN_Atom_Scalar_Head(nn.Module, HeadInterface):
    """
    Head for per-atom scalar properties.
    Applies node-level MLP to node features.
    """
    def __init__(self, 
                 backbone: CGCNNBackbone, 
                 output_property: str = "charges", 
                 h_fea_len: int = 128, 
                 n_h: int = 1,
                 **kwargs):
        super().__init__()
        self.output_property = output_property
        self.conv_to_fc = nn.Linear(backbone.atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.n_h = n_h
        
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h-1)])
            
        self.fc_out = nn.Linear(h_fea_len, 1)

    def forward(self, data_dict: AtomicData, emb: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        atom_fea = emb["node_features"]
        
        # Fully connected layers per node
        out = self.conv_to_fc(self.conv_to_fc_softplus(atom_fea))
        out = self.conv_to_fc_softplus(out)
        if self.n_h > 1:
            for fc, softplus in zip(self.fcs, self.softpluses):
                out = softplus(fc(out))
                
        out = self.fc_out(out)
        
        # Mask out pseudo-nodes if atom_weight exists (set prediction to 0)
        atom_weight = data_dict.get("atom_weight", None)
        if atom_weight is not None:
            out = out * atom_weight.unsqueeze(-1)
            
        return {self.output_property: out.squeeze(-1) if out.shape[-1] == 1 else out}
