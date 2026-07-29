import torch
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch

def test_batch_collation():
    print("Running Batch Collation Test...")
    
    # Graph 1: 5 atoms, cluster partition has 2 atoms (indices 0, 1)
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
    data1.atom_weight = torch.tensor([2.0, 1.0, 0.0, 0.0, 0.0])
    
    # Graph 2: 10 atoms, cluster partition has 3 atoms (indices 0, 2, 4)
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
    data2.atom_weight = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Batch them
    batch = atomicdata_list_to_batch([data1, data2])
    
    print(f"Batch natoms: {batch.natoms}")
    print(f"Batch cluster_partition: {batch.cluster_partition.tolist()}")
    print(f"Batch atom_weight shape: {batch.atom_weight.shape}")
    print(f"Batch atom_weight values: {batch.atom_weight.tolist()}")
    
    # Assertions
    expected_cluster_partition = [0, 1, 5, 7, 9] # data2 indices shifted by natoms of data1 (5)
    assert batch.cluster_partition.tolist() == expected_cluster_partition, f"Error in cluster_partition: {batch.cluster_partition.tolist()} != {expected_cluster_partition}"
    
    expected_weight_length = 15
    assert batch.atom_weight.shape[0] == expected_weight_length, f"Error in atom_weight length: {batch.atom_weight.shape[0]} != {expected_weight_length}"
    
    print("Batch Collation Test Passed successfully!\n")

if __name__ == "__main__":
    test_batch_collation()
