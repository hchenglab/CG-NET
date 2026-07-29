"""
Generates an ASE LMDB dataset from OC25 raw ASELMDB trajectories for Force training.
Supports tag re-assignment and spatial truncation identically to HEA script.
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
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_global_config = None

def _init_processing_pool(config: dict) -> None:
    global _global_config
    _global_config = config

def process_file(mp_arg: tuple) -> tuple[str, list[int], int, int]:
    in_file, out_file, row_ids, worker_id = mp_arg
    natoms_list = []
    successful = 0
    failed = 0

    try:
        with connect(str(out_file)) as out_db:
            with connect(str(in_file)) as in_db:
                for r_id in tqdm(row_ids, position=worker_id, desc=f"Worker {worker_id}"):
                    try:
                        row = in_db.get(id=r_id)
                        atoms = row.toatoms()
                        
                        if atoms.info is None:
                            atoms.info = {}
                        
                        # 1. Identify adsorbate for cluster truncation
                        # OC25 original tags: 0 for fixed (subsurface), 1 for surface, 2 for adsorbate, 3 for solvent
                        old_tags = atoms.get_tags()
                        is_adsorbate = (old_tags == 2)
                        
                        # Filter out structures with no adsorbate (bare slabs) uniformly for all modes
                        if not is_adsorbate.any():
                            logger.warning(f"Skipping row {row.id} in {in_file}: no adsorbate atoms found (bare slab).")
                            failed += 1
                            continue
                        
                        # 2. Add forces to info
                        has_forces = False
                        try:
                            forces = atoms.get_forces()
                            atoms.arrays["forces"] = forces
                            has_forces = True
                            if "forces" in atoms.info:
                                del atoms.info["forces"]
                        except Exception:
                            if "forces" in atoms.arrays:
                                has_forces = True
                            elif "forces" in atoms.info:
                                atoms.arrays["forces"] = np.array(atoms.info["forces"])
                                del atoms.info["forces"]
                                has_forces = True

                        if not has_forces:
                            logger.warning(f"No forces found for id {row.id} in {in_file}, skipping.")
                            failed += 1
                            continue
                            
                        # Store energy
                        try:
                            atoms.info["energy"] = atoms.get_potential_energy()
                        except Exception:
                            pass
                            
                        # Extract any key-value pairs
                        if hasattr(row, 'key_value_pairs'):
                            for k, v in row.key_value_pairs.items():
                                atoms.info[k] = v
                        if hasattr(row, 'data') and row.data:
                            for k, v in row.data.items():
                                atoms.info[k] = v

                        # 3. Cluster Truncation
                        truncation_mode = _global_config.get("truncation_mode", "none")
                        cluster_radius = _global_config.get("cluster_radius")
                        
                        if is_adsorbate.any():
                            ads_indices = np.where(is_adsorbate)[0]
                            anchor = ads_indices[0]
                            vecs = atoms.get_distances(anchor, ads_indices, mic=True, vector=True)
                            pos = atoms.get_positions()
                            pos[ads_indices] = pos[anchor] + vecs
                            cluster_center = pos[ads_indices].mean(axis=0)
                            
                            # Strategy A: Filter if adsorbates are too scattered
                            max_ads_dist = np.linalg.norm(pos[ads_indices] - cluster_center, axis=1).max()
                            max_ads_spread = _global_config.get("max_ads_spread", 4.0)
                            if max_ads_spread is not None and max_ads_dist > max_ads_spread:
                                logger.warning(f"Skipping row {row.id} in {in_file}: adsorbate max distance {max_ads_dist:.2f} > {max_ads_spread}")
                                failed += 1
                                continue
                        else:
                            cluster_center = atoms.get_positions().mean(axis=0)
                            
                        if truncation_mode in ["local", "local-convention"] and cluster_radius is not None:
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
                                for key in atoms.arrays:
                                    if key not in ["positions", "numbers"]:
                                        supercell.arrays[key] = np.tile(atoms.arrays[key], (M1 * M2 * M3,) + (1,) * (atoms.arrays[key].ndim - 1))
                                
                                supercell.pbc = [False, False, False]
                                
                                translation = rep_a1 * a1 + rep_a2 * a2 + rep_a3 * a3
                                new_cluster_center = cluster_center + translation
                                supercell.info["cluster_center"] = new_cluster_center.tolist()
                                
                                distances = np.linalg.norm(supercell.get_positions() - new_cluster_center, axis=1)
                                target_indices = np.where(distances <= cluster_radius)[0]
                                retained_indices = np.where(distances <= max_radius)[0]
                                
                                if len(retained_indices) == 0:
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

                        # Restore forces to info
                        if "forces" in atoms.arrays:
                            atoms.info["forces"] = atoms.arrays["forces"].tolist()
                            del atoms.arrays["forces"]
                            
                        out_db.write(atoms, data=atoms.info)
                        natoms_list.append(len(atoms))
                        successful += 1

                    except Exception as e:
                        logger.error(f"Error processing row {row.id} in {in_file}: {e}")
                        failed += 1

    except Exception as e:
        logger.error(f"Error opening DB files {in_file} -> {out_file}: {e}")

    return out_file, natoms_list, successful, failed


def process_split(split_name: str, input_dir: str, output_dir: str, num_workers: int):
    split_in_dir = os.path.join(input_dir, split_name)
    split_out_dir = os.path.join(output_dir, split_name)
    
    if not os.path.exists(split_in_dir):
        logger.info(f"Split dir {split_in_dir} not found. Skipping {split_name}.")
        return

    os.makedirs(split_out_dir, exist_ok=True)
    
    in_files = glob.glob(os.path.join(split_in_dir, "data*.aselmdb"))
    in_files.sort()
    
    if not in_files:
        logger.info(f"No aselmdb files found in {split_in_dir}")
        return
        
    all_items = []
    logger.info(f"Scanning lengths of {len(in_files)} files in {split_name}...")
    for in_file in in_files:
        with connect(str(in_file)) as in_db:
            length = len(in_db)
            for r_id in range(1, length + 1):
                all_items.append((in_file, r_id))
                
    num_samples = _global_config.get(f"num_{split_name}_samples", None)
    if num_samples is not None and num_samples < len(all_items):
        logger.info(f"Sampling {num_samples} items out of {len(all_items)} in {split_name}...")
        # Use deterministic integer addition instead of hash() which is randomized per python process!
        split_offset = {"train": 1, "val": 2, "test": 3}.get(split_name, 0)
        rng = random.Random(_global_config["seed"] + split_offset)
        all_items = rng.sample(all_items, num_samples)
        
    from collections import defaultdict
    file_to_row_ids = defaultdict(list)
    for in_file, r_id in all_items:
        file_to_row_ids[in_file].append(r_id)
        
    mp_args = []
    for i, in_file in enumerate(in_files):
        if in_file not in file_to_row_ids:
            continue
            
        row_ids = sorted(file_to_row_ids[in_file])
        basename = os.path.basename(in_file)
        out_file = os.path.join(split_out_dir, basename)
        mp_args.append((in_file, out_file, row_ids, i % num_workers))
        
    with mp.Pool(num_workers) as pool:
        results = pool.map(process_file, mp_args)
        
    results.sort(key=lambda x: str(x[0]))
    
    total_natoms = []
    total_success = 0
    total_failed = 0
    
    for r in results:
        total_natoms.extend(r[1])
        total_success += r[2]
        total_failed += r[3]
        
    np.savez_compressed(os.path.join(split_out_dir, "metadata.npz"), natoms=total_natoms)
    logger.info(f"Split {split_name}: generated {total_success} frames, failed {total_failed} structures")


def main():
    parser = argparse.ArgumentParser(description="Generate OC25 Force Dataset")
    parser.add_argument("--input-dir", required=True, type=str, help="Path to raw OC25 dataset directory (contains train/val)")
    parser.add_argument("--output-dir", required=True, type=str, help="Output directory for processed ASE DBs")
    parser.add_argument("--truncation-mode", choices=["none", "local", "local-convention"], default="none")
    parser.add_argument("--cluster-radius", type=float, default=6.0)
    parser.add_argument("--pseudo-radius", type=float, default=6.0)
    parser.add_argument("--max-ads-spread", type=float, default=4.0, help="Maximum allowed distance from cluster center to any adsorbate atom")
    parser.add_argument("--with-pseudonodes", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-train-samples", type=int, default=None, help="Number of samples to draw from train split")
    parser.add_argument("--num-val-samples", type=int, default=None, help="Number of samples to draw from val split")
    
    args = parser.parse_args()
    
    _init_processing_pool({
        "truncation_mode": args.truncation_mode,
        "cluster_radius": args.cluster_radius,
        "pseudo_radius": args.pseudo_radius,
        "max_ads_spread": args.max_ads_spread,
        "cluster_with_pseudonodes": args.with_pseudonodes,
        "seed": args.seed,
        "num_train_samples": args.num_train_samples,
        "num_val_samples": args.num_val_samples
    })
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    for split in ["train", "val"]:
        process_split(split, args.input_dir, args.output_dir, args.num_workers)
        
    logger.info("Dataset generation complete!")

if __name__ == "__main__":
    main()
