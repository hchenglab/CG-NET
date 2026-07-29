import torch
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import eSCNMDBackbone, Linear_System_Scalar_Head
from fairchem.core.common.registry import registry
from torch.optim import Adam

def create_dummy_data():
    # Graph 1: 5 atoms, target scalar = 2.5
    data1 = AtomicData(
        pos=torch.rand(5, 3),
        atomic_numbers=torch.tensor([1, 6, 1, 1, 1]),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
        natoms=torch.tensor([5]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        cell_offsets=torch.zeros((0, 3)),
        nedges=torch.tensor([0]),
        charge=torch.tensor([0.0]),
        spin=torch.tensor([0.0]),
        fixed=torch.zeros(5, dtype=torch.long),
        tags=torch.zeros(5, dtype=torch.long),
        batch=torch.zeros(5, dtype=torch.long)
    )
    data1.cluster_partition = torch.tensor([0, 1], dtype=torch.long)
    data1.atom_weight = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    data1.target = torch.tensor([2.5])
    
    # Graph 2: 10 atoms, target scalar = -1.0
    data2 = AtomicData(
        pos=torch.rand(10, 3),
        atomic_numbers=torch.tensor([6, 6, 8, 1, 1, 1, 1, 1, 1, 1]),
        cell=torch.eye(3).unsqueeze(0),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
        natoms=torch.tensor([10]),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        cell_offsets=torch.zeros((0, 3)),
        nedges=torch.tensor([0]),
        charge=torch.tensor([0.0]),
        spin=torch.tensor([0.0]),
        fixed=torch.zeros(10, dtype=torch.long),
        tags=torch.zeros(10, dtype=torch.long),
        batch=torch.zeros(10, dtype=torch.long)
    )
    data2.cluster_partition = torch.tensor([0, 2, 4], dtype=torch.long)
    data2.atom_weight = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    data2.target = torch.tensor([-1.0])
    
    return [data1, data2]

def run_overfit_test():
    print("Setting up micro-dataset...")
    data_list = create_dummy_data()
    batch = atomicdata_list_to_batch(data_list)
    
    print("Initializing model...")
    # Instantiate backbone with very small parameters for speed
    backbone = eSCNMDBackbone(
        num_layers=1,
        max_neighbors=20,
        cutoff=5.0,
        sphere_channels=16,
        hidden_channels=16,
        lmax=1,
        mmax=1,
        distance_function="gaussian",
        regress_forces=False,
        regress_stress=False,
        use_dataset_embedding=False,
    )
    
    head = Linear_System_Scalar_Head(
        backbone=backbone,
        output_property="scalar",
        reduce="weighted_mean"
    )
    
    # Optional: If you want to use the Head alongside the backbone
    # (Here we manually orchestrate the forward pass for simplicity)
    
    optimizer = Adam(list(backbone.parameters()) + list(head.parameters()), lr=0.01)
    loss_fn = torch.nn.MSELoss()
    
    print("Starting training loop...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        
        # 1. Forward backbone
        # We can pass `batch` directly since it's an AtomicData object that acts like a dict
        emb = backbone(batch)
        
        # 2. Forward head
        outputs = head(batch, emb)
        predictions = outputs["scalar"]
        
        # 3. Calculate loss
        target = torch.cat([d.target for d in data_list]).to(predictions.dtype)
        loss = loss_fn(predictions, target)
        
        # 4. Backward
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.6f} | Preds: {predictions.detach().tolist()}")
            
    print("\nOverfitting Test Finished!")
    if loss.item() < 0.1:
        print("✅ SUCCESS: The model successfully overfit to the micro-dataset, confirming data flows correctly.")
    else:
        print("❌ FAILED: The loss did not drop. Check your broadcast shapes or learning rates.")

if __name__ == "__main__":
    run_overfit_test()
