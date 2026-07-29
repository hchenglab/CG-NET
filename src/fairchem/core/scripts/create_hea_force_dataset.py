"""
Generates an ASE LMDB dataset from HEA adsorption trajectories for Force training.
Supports random sampling of frames, tag re-assignment, and spatial truncation.
"""

import argparse
import glob
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_global_config = None

def _init_processing_pool(config: dict) -> None:
    global _global_config
    _global_config = config

def process_file_list(mp_arg: tuple) -> tuple[str, list[int], int, int]:
    db_file, traj_files, worker_id = mp_arg
    natoms_list = []
    successful = 0
    failed = 0

    with connect(str(db_file)) as db:
        for traj_file in tqdm(traj_files, position=worker_id, desc=f"Worker {worker_id}"):
            try:
                base_name = os.path.basename(traj_file).replace(".traj", "")
                
                # 1. Parse structures from traj file
                atoms_list = read(traj_file, index=":")
                if not atoms_list:
                    logger.warning(f"No atoms in {traj_file}")
                    failed += 1
                    continue
                
                # 2. Sample N frames uniformly from the trajectory to ensure coverage (including final state)
                num_samples = _global_config.get("num_samples", 3)
                total_frames = len(atoms_list)
                if total_frames <= num_samples:
                    selected_frames = atoms_list
                else:
                    indices = np.linspace(0, total_frames - 1, num=num_samples, dtype=int)
                    selected_frames = [atoms_list[i] for i in indices]
                
                for frame_idx, atoms in enumerate(selected_frames):
                    if atoms.info is None:
                        atoms.info = {}
                    
                    atoms.info["system_id"] = f"{base_name}_frame{frame_idx}"
                    
                    # Store energy if available, but for direct force training it won't be used as loss
                    try:
                        atoms.info["energy"] = atoms.get_potential_energy()
                    except Exception:
                        pass
                        
                    # 3. Robustly extract forces
                    has_forces = False
                    try:
                        forces = atoms.get_forces()
                        atoms.arrays["forces"] = forces
                        has_forces = True
                        if "forces" in atoms.info:
                            del atoms.info["forces"]
                    except Exception:
                        if "forces" in atoms.info:
                            atoms.arrays["forces"] = np.array(atoms.info["forces"])
                            del atoms.info["forces"]
                            has_forces = True

                    if not has_forces:
                        logger.warning(f"No forces found in {traj_file}, skipping.")
                        failed += 1
                        continue

                    # 4. Redefine Tags
                    # Original: 0=adsorbate, 1/2/3/4=layers
                    # New: 0=fixed, 1=surface, 2=adsorbate
                    old_tags = atoms.get_tags()
                    fixed_indices = atoms.constraints[0].index if atoms.constraints else []
                    
                    is_adsorbate = (old_tags == 0)
                    is_fixed = np.isin(np.arange(len(atoms)), fixed_indices)
                    is_surface = (~is_fixed) & (~is_adsorbate)
                    
                    new_tags = np.zeros_like(old_tags)
                    new_tags[is_fixed] = 0
                    new_tags[is_surface] = 1
                    new_tags[is_adsorbate] = 2
                    atoms.set_tags(new_tags)
                    
                    # 5. Cluster Truncation
                    truncation_mode = _global_config.get("truncation_mode", "none")
                    cluster_radius = _global_config.get("cluster_radius")
                    
                    # Calculate cluster center universally
                    if is_adsorbate.any():
                        ads_indices = np.where(is_adsorbate)[0]
                        anchor = ads_indices[0]
                        vecs = atoms.get_distances(anchor, ads_indices, mic=True, vector=True)
                        pos = atoms.get_positions()
                        pos[ads_indices] = pos[anchor] + vecs
                        cluster_center = pos[ads_indices].mean(axis=0)
                    else:
                        cluster_center = atoms.get_positions().mean(axis=0)
                        
                    if truncation_mode in ["local", "local-convention"] and cluster_radius is not None:
                        if not is_adsorbate.any():
                            logger.warning(f"No adsorbate atoms found in {traj_file}")
                            failed += 1
                            continue
                            
                        atoms.info["cluster_center"] = cluster_center.tolist()
                        
                        from pymatgen.io.ase import AseAtomsAdaptor
                        struct = AseAtomsAdaptor.get_structure(atoms)
                        
                        if truncation_mode == "local":
                            target_sites = struct.get_sites_in_sphere(cluster_center.tolist(), cluster_radius)
                            target_indices = [s.index for s in target_sites]
                            target_weight_full = np.bincount(target_indices, minlength=len(atoms))
                            target_partition_np = np.unique(target_indices)
                            
                            if _global_config.get("cluster_with_pseudonodes", False):
                                radius = _global_config.get("pseudo_radius", 6.0)
                                all_sites = struct.get_sites_in_sphere(cluster_center.tolist(), cluster_radius + radius)
                                retained_indices = np.unique([s.index for s in all_sites])
                            else:
                                retained_indices = target_partition_np
                                
                            if len(retained_indices) == 0:
                                logger.warning(f"Skipping {base_name}: retained_indices is empty in local mode.")
                                failed += 1
                                continue
                                
                            atoms = atoms[retained_indices]
                            
                            atoms.info["atom_weight"] = target_weight_full[retained_indices].tolist()
                            
                            old_to_new = {old: new for new, old in enumerate(retained_indices)}
                            new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
                            atoms.info["cluster_partition"] = new_target_partition
                            
                        elif truncation_mode == "local-convention":
                            if _global_config.get("cluster_with_pseudonodes", False):
                                max_radius = cluster_radius + _global_config.get("pseudo_radius", 6.0)
                            else:
                                max_radius = cluster_radius
                                
                            vol = atoms.get_volume()
                            a1, a2, a3 = atoms.cell
                            if vol > 1e-6:
                                rep_a1 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a2, a3)) / vol))
                                rep_a2 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a3, a1)) / vol))
                                rep_a3 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a1, a2)) / vol))
                            else:
                                rep_a1 = rep_a2 = rep_a3 = 0
                            M1, M2, M3 = 2 * rep_a1 + 1, 2 * rep_a2 + 1, 2 * rep_a3 + 1
                            
                            supercell = atoms * [M1, M2, M3]
                            supercell.info = atoms.info.copy()
                            # Restore arrays correctly across supercell
                            for key in atoms.arrays:
                                if key not in ["positions", "numbers"]:
                                    supercell.arrays[key] = np.tile(atoms.arrays[key], (M1 * M2 * M3,) + (1,) * (atoms.arrays[key].ndim - 1))
                            
                            supercell.pbc = [False, False, False]
                            
                            # Shift cluster center to the middle unit cell
                            translation = rep_a1 * a1 + rep_a2 * a2 + rep_a3 * a3
                            new_cluster_center = cluster_center + translation
                            supercell.info["cluster_center"] = new_cluster_center.tolist()
                            
                            # Physical truncation around the shifted cluster_center
                            distances = np.linalg.norm(supercell.get_positions() - new_cluster_center, axis=1)
                            target_indices = np.where(distances <= cluster_radius)[0]
                            retained_indices = np.where(distances <= max_radius)[0]
                            
                            if len(retained_indices) == 0:
                                logger.warning(f"Skipping {base_name}: retained_indices is empty in local-convention mode.")
                                failed += 1
                                continue
                                
                            target_weight_full = np.bincount(target_indices, minlength=len(supercell))
                            target_partition_np = target_indices
                            
                            supercell = supercell[retained_indices]
                            supercell.info["atom_weight"] = target_weight_full[retained_indices].tolist()
                            
                            old_to_new = {old: new for new, old in enumerate(retained_indices)}
                            new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
                            supercell.info["cluster_partition"] = new_target_partition
                            
                            atoms = supercell
                            
                    else:
                        atoms.info["cluster_center"] = cluster_center.tolist()
                        
                        if cluster_radius is not None:
                            vol = atoms.get_volume()
                            if vol > 1e-6:
                                from pymatgen.io.ase import AseAtomsAdaptor
                                struct = AseAtomsAdaptor.get_structure(atoms)
                                target_sites = struct.get_sites_in_sphere(cluster_center.tolist(), cluster_radius)
                                target_partition_np = np.unique([s.index for s in target_sites])
                            else:
                                distances = np.linalg.norm(atoms.get_positions() - cluster_center, axis=1)
                                target_partition_np = np.where(distances <= cluster_radius)[0]
                                
                            atoms.info["cluster_partition"] = target_partition_np.tolist()
                            
                            target_weight_full = np.bincount(target_partition_np, minlength=len(atoms))
                            atoms.info["atom_weight"] = target_weight_full.tolist()

                    # Restore forces to info if they were preserved in arrays
                    if "forces" in atoms.arrays:
                        atoms.info["forces"] = atoms.arrays["forces"].tolist()
                        del atoms.arrays["forces"]
                        
                    db.write(atoms, data=atoms.info)
                    natoms_list.append(len(atoms))
                    successful += 1

            except Exception as e:
                logger.error(f"Error processing {traj_file}: {e}")
                failed += 1

    return db_file, natoms_list, successful, failed

def split_and_create_dbs(files, output_dir, prefix, split_ratios, num_workers, chunk_size):
    random.shuffle(files)
    n_total = len(files)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])
    
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }
    
    for split_name, split_files in splits.items():
        if not split_files:
            continue
            
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        chunks = [split_files[i:i + chunk_size] for i in range(0, len(split_files), chunk_size)]
        mp_args = [
            (os.path.join(split_dir, f"data.{i:04d}.aselmdb"), chunk, i)
            for i, chunk in enumerate(chunks)
        ]
        
        with mp.Pool(num_workers) as pool:
            results = pool.map(process_file_list, mp_args)
            
        # Re-sort results to guarantee deterministic ordering matching the input order
        results.sort(key=lambda x: str(x[0]))
            
        total_natoms = []
        for r in results:
            total_natoms.extend(r[1])
            
        total_success = sum(r[2] for r in results)
        total_failed = sum(r[3] for r in results)
        
        np.savez_compressed(os.path.join(split_dir, "metadata.npz"), natoms=total_natoms)
        
        logger.info(f"Split {split_name}: generated {total_success} frames, failed {total_failed} files")

def main():
    parser = argparse.ArgumentParser(description="Generate HEA Force Dataset")
    parser.add_argument("--input-dir", required=True, type=str, help="Path to HEA traj directory")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for ASE DBs")
    parser.add_argument("--truncation-mode", choices=["none", "local", "local-convention"], default="none")
    parser.add_argument("--cluster-radius", type=float, default=6.0)
    parser.add_argument("--pseudo-radius", type=float, default=6.0)
    parser.add_argument("--with-pseudonodes", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=3, help="Number of frames to sample per trajectory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    
    args = parser.parse_args()
    
    _init_processing_pool({
        "truncation_mode": args.truncation_mode,
        "cluster_radius": args.cluster_radius,
        "pseudo_radius": args.pseudo_radius,
        "cluster_with_pseudonodes": args.with_pseudonodes,
        "num_samples": args.num_samples
    })
    
    traj_pattern = os.path.join(args.input_dir, "hea_*_ads*.traj")
    traj_files = glob.glob(traj_pattern)
    # Ignore pure slab files
    traj_files = [f for f in traj_files if "slab" not in f]
    
    # Sort files to guarantee deterministic order across different file systems
    traj_files.sort()
    
    logger.info(f"Found {len(traj_files)} HEA adsorbate traj files.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seeds for reproducible splits
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Ratios: 8:1:1
    split_ratios = [0.8, 0.1, 0.1]
    
    split_and_create_dbs(traj_files, args.output_dir, "hea_force", split_ratios, args.num_workers, args.chunk_size)
    logger.info("Dataset generation complete!")

if __name__ == "__main__":
    main()
