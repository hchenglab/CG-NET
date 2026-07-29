import os
import pytest
import numpy as np
from pathlib import Path
from fairchem.core.scripts.create_oc20_dataset import launch_processing
from fairchem.core.datasets import AseDBDataset

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_oc20_dataset_generation_correctness(tmp_path):
    """
    Tests the correctness of create_oc20_dataset.py by running it on a 
    small subset of the actual OC20 dataset on disk.
    
    Verifies:
    1. metadata.npz is correctly generated and natoms are perfectly ordered.
    2. cluster_center is dynamically calculated using tags == 2.
    3. adsorption_energy is correctly populated.
    """
    input_dir = "/Users/chenghao/research/datasets/oc20/s2ef/train/200K/s2ef_train_200K"
    mapping_file = "/Users/chenghao/research/datasets/oc20/oc20_data_mapping.pkl"
    
    if not os.path.exists(input_dir) or not os.path.exists(mapping_file):
        pytest.skip(f"OC20 dataset not found locally at {input_dir}")
        
    mock_input_dir = tmp_path / "input"
    mock_input_dir.mkdir()
    
    # Symlink only the first two file pairs to save time but test multiprocessing
    for i in [0, 1]:
        extxyz_file = f"{input_dir}/{i}.extxyz.xz"
        txt_file = f"{input_dir}/{i}.txt.xz"
        if os.path.exists(extxyz_file) and os.path.exists(txt_file):
            os.symlink(extxyz_file, mock_input_dir / f"{i}.extxyz.xz")
            os.symlink(txt_file, mock_input_dir / f"{i}.txt.xz")
        
    output_dir = tmp_path / "output"
    
    # 1. Run the dataset creation script with 2 workers to test multiprocessing aggregation
    launch_processing(
        input_dirs=[str(mock_input_dir)],
        output_dir=str(output_dir),
        num_workers=2,
        mapping_file=mapping_file
    )
    
    # 2. Verify metadata.npz exists and natoms is correctly ordered
    metadata_path = output_dir / "metadata.npz"
    assert metadata_path.exists(), "metadata.npz was not created"
    
    metadata = np.load(metadata_path)
    assert "natoms" in metadata, "'natoms' missing from metadata.npz"
    total_natoms = metadata["natoms"]
    
    # 3. Verify AseDBDataset can read it and the size matches metadata
    db_pattern = str(output_dir / "data.*.aselmdb")
    dataset = AseDBDataset({"src": db_pattern})
    
    assert len(dataset) > 0, "Dataset is empty"
    assert len(dataset) == len(total_natoms), "Length of dataset does not match metadata natoms"
    
    # 4. Verify the correctness of individual atoms properties
    # We check the first few frames to ensure the alignment and calculations are correct
    for idx in range(min(100, len(dataset))):
        atoms = dataset.get_atoms(idx)
        
        # A. Check natoms alignment
        assert len(atoms) == total_natoms[idx], f"Misalignment at idx {idx}: atoms len {len(atoms)} != natoms {total_natoms[idx]}"
        
        # B. Check adsorption_energy
        assert "adsorption_energy" in atoms.info, "adsorption_energy not in atoms.info"
        assert isinstance(atoms.info["adsorption_energy"], float), "adsorption_energy is not a float"
        
        # C. Check dynamic cluster_center logic
        assert "cluster_center" in atoms.info, "cluster_center not in atoms.info"
        cluster_center = np.array(atoms.info["cluster_center"])
        
        tags = atoms.get_tags()
        if (tags == 2).any():
            # If adsorbate is present, cluster_center MUST match its center of mass
            expected_center = atoms.get_positions()[tags == 2].mean(axis=0)
            np.testing.assert_allclose(
                cluster_center, 
                expected_center, 
                rtol=1e-5, 
                atol=1e-5,
                err_msg=f"cluster_center does not dynamically track adsorbate at idx {idx}"
            )

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        print(f"Running test in {tmpdirname}...")
        test_oc20_dataset_generation_correctness(Path(tmpdirname))
        print("Test passed successfully!")
