import numpy as np
from ase import Atoms
from ase.io import write

def create_dummy_dataset():
    structures = []
    for _ in range(5):
        # 10 random atoms
        pos = np.random.rand(10, 3) * 5.0
        atomic_numbers = np.random.randint(1, 10, size=10)
        
        # A cluster partition is usually a list of indices. We can just store them as arrays.
        # But let's actually just create simple atoms and rely on the config to pass 
        # `cluster_center` and `cluster_radius` to test that specific logic!
        # If we want to hardcode them, we can do:
        atoms = Atoms(numbers=atomic_numbers, positions=pos, cell=[10, 10, 10], pbc=True)
        
        # We want to predict a scalar. The head uses `energy` as the default target.
        # Let's set a random energy.
        atoms.info["energy"] = np.random.rand() * 10.0
        
        structures.append(atoms)
        
    write("dummy_dataset.extxyz", structures)
    print("Created dummy_dataset.extxyz with 5 structures.")

if __name__ == "__main__":
    create_dummy_dataset()
