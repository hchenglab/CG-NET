import torch
import torch.nn as nn
from fairchem.core.datasets.atomic_data import AtomicData

# Try importing compute_energy depending on the actual import path
from fairchem.core.models.uma.outputs import compute_energy
from fairchem.core.models.uma.escn_md import Linear_System_Scalar_Head
from unittest.mock import MagicMock

def test_gradient_isolation():
    print("Running Gradient and Isolation Test...")
    
    # 1. Mock the eSCNMDBackbone since we only want to test the Head and outputs.py
    mock_backbone = MagicMock()
    mock_backbone.sphere_channels = 16
    
    # Instantiate the Linear_System_Scalar_Head
    # NOTE: Set reduce="weighted_mean" to test if atom_weight properly masks non-cluster atoms
    head = Linear_System_Scalar_Head(mock_backbone, output_property="scalar", reduce="weighted_mean")
    
    # 2. Create mock data simulating a batch
    # Let's say we have 1 graph with 5 atoms.
    # The cluster only contains atoms 0 and 1.
    natoms = 5
    pos = torch.rand(natoms, 3, requires_grad=True)
    
    # We simulate atom_weight: atoms 0 and 1 have weight 1.0, others have 0.0
    atom_weight = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    data_dict = {
        "batch": torch.zeros(natoms, dtype=torch.long),
        "natoms": torch.tensor([natoms]),
        "atom_weight": atom_weight
    }
    
    # 3. Simulate node embeddings coming out of the backbone
    # Shape of emb["node_embedding"]: [N, (lmax+1)^2, C]
    # Here we mock it for L=0 only for simplicity: shape [N, 1, sphere_channels]
    mock_node_embedding = torch.randn(natoms, 1, mock_backbone.sphere_channels, requires_grad=True)
    
    # We assume pos affects node embeddings (to test autograd flow)
    # mock_node_embedding = mock_node_embedding_base + pos.sum(dim=-1, keepdim=True).unsqueeze(-1)
    # Since we are testing isolation, we create an explicit dependency
    emb = {
        "node_embedding": mock_node_embedding + pos.mean(dim=-1).view(-1, 1, 1).expand(-1, 1, mock_backbone.sphere_channels)
    }
    
    # 4. Forward Pass
    predictions = head(data_dict, emb)
    system_scalar = predictions["scalar"]
    
    print(f"Predicted scalar: {system_scalar.item():.4f}")
    
    # 5. Backward Pass
    system_scalar.backward()
    
    # 6. Check Gradients
    print("\nPos gradients (should be 0.0 for atoms not in cluster, i.e., index 2, 3, 4):")
    for i in range(natoms):
        grad_norm = pos.grad[i].norm().item() if pos.grad[i] is not None else 0.0
        print(f"Atom {i} (weight={atom_weight[i].item()}): gradient norm = {grad_norm:.6f}")
        
        # Validation
        if atom_weight[i].item() == 0.0:
            assert grad_norm < 1e-7, f"ERROR: Atom {i} has weight 0 but received non-zero gradient {grad_norm}!"
        else:
            assert grad_norm > 1e-7, f"ERROR: Atom {i} has weight > 0 but received zero gradient!"

    print("\nGradient Test Passed: Isolation is functioning correctly when reduce='weighted_mean'!")

if __name__ == "__main__":
    test_gradient_isolation()
