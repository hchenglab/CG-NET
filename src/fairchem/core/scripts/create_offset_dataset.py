"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Converts .traj dataset into ASE LMDB database format suitable for UMA model training and evaluation.
Supports an experimental --center-offset flag to shift the local graph center by a distance d
to test the model's sensitivity to cluster center selection.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp
import os
import random
import csv
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for multiprocessing
_global_mapping = None
_global_config = None


def _init_processing_pool(mapping: dict, config: dict) -> None:
    """
    Initialize global mapping dictionary and config for worker processes.
    """
    global _global_mapping, _global_config
    _global_mapping = mapping
    _global_config = config


def compute_adsorption_energy_rms(
    train_path: str | Path,
) -> float:
    """
    Compute the adsorption_energy_rms from the generated training LMDB dataset.
    """
    train_path = Path(train_path)
    db_pattern = str(train_path / "data.*.aselmdb")

    dataset = AseDBDataset({"src": db_pattern})
    dataset_size = len(dataset)

    energies = []
    for idx in tqdm(range(dataset_size), desc="Computing normalizer values"):
        atoms = dataset.get_atoms(idx)
        energies.append(atoms.info["adsorption_energy"])

    energies = np.array(energies)
    energy_rms = np.sqrt(np.mean(np.square(energies)))

    assert np.isfinite(energy_rms).all(), "Found non-finite values in adsorption energies!"

    return float(energy_rms)


def process_file_list(mp_arg: tuple) -> tuple[str, list, list, list]:
    """
    Process a chunk of .traj files assigned to a worker and write to ASE DB.
    """
    db_file, traj_files, worker_id = mp_arg

    successful = []
    failed = []
    natoms = []

    # Use a worker-specific seed so random offsets are diverse and reproducible
    np.random.seed(worker_id + _global_config.get("seed", 42))

    with connect(str(db_file)) as db:
        for traj_file in tqdm(traj_files, position=worker_id, desc=f"Worker {worker_id}"):
            try:
                base_name = os.path.basename(traj_file).replace(".traj", "")
                
                if base_name not in _global_mapping:
                    failed.append(f"{os.path.basename(traj_file)}: Not found in mapping CSV")
                    continue
                    
                metadata = _global_mapping[base_name]

                # Parse structures from traj file
                atoms_list = read(traj_file, index=":")
                
                if not atoms_list:
                    failed.append(f"{os.path.basename(traj_file)}: No atoms found in traj")
                    continue
                
                # Use the last frame
                atoms = atoms_list[-1]
                
                # Set metadata
                if atoms.info is None:
                    atoms.info = {}
                
                atoms.info["system_id"] = base_name
                atoms.info["adsorption_energy"] = metadata["adsorption_energy"]
                atoms.info["energy"] = metadata["adsorption_energy"]
                atoms.info["cidxs"] = metadata["cidxs"]
                
                # Calculate default cluster center
                tags = atoms.get_tags()
                ads_mask = (tags == 2)
                if ads_mask.any():
                    ads_indices = np.where(ads_mask)[0]
                    anchor = ads_indices[0]
                    vecs = atoms.get_distances(anchor, ads_indices, mic=True, vector=True)
                    pos = atoms.get_positions()
                    pos[ads_indices] = pos[anchor] + vecs
                    cluster_center = pos[ads_indices].mean(axis=0)
                else:
                    cidx = int(metadata["cidxs"])
                    if 0 <= cidx < len(atoms):
                        cluster_center = atoms.get_positions()[cidx]
                    else:
                        raise ValueError(f"cidxs {cidx} out of bounds for atoms with length {len(atoms)}")

                # --- NEW LOGIC: OFFSET THE CENTER ---
                offset_dist = _global_config.get("center_offset", 0.0)
                if offset_dist > 0.0:
                    offset_mode = _global_config.get("offset_mode", "xyz")
                    if offset_mode == "xy":
                        # Random direction in XY plane only (useful for 2D impurities/surfaces)
                        angle = np.random.uniform(0, 2 * np.pi)
                        vec = np.array([np.cos(angle), np.sin(angle), 0.0])
                    else:
                        # Random direction in 3D
                        vec = np.random.randn(3)
                        vec /= np.linalg.norm(vec)
                    
                    cluster_center += vec * offset_dist
                # ------------------------------------
                        
                truncation_mode = _global_config.get("truncation_mode", "none")
                cluster_radius = _global_config.get("cluster_radius")

                if truncation_mode == "local" and cluster_radius is not None:
                    from pymatgen.io.ase import AseAtomsAdaptor
                    struct = AseAtomsAdaptor.get_structure(atoms)
                    
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
                    
                    try:
                        forces = atoms.get_forces()
                        atoms.info["forces"] = forces[retained_indices]
                    except Exception:
                        if "forces" in atoms.info:
                            atoms.info["forces"] = np.array(atoms.info["forces"])[retained_indices]
                        
                    if len(retained_indices) == 0:
                        logger.warning(f"Skipping {base_name}: retained_indices is empty in local mode.")
                        continue
                    atoms = atoms[retained_indices]
                    
                    atoms.info["atom_weight"] = target_weight_full[retained_indices].tolist()
                    
                    old_to_new = {old: new for new, old in enumerate(retained_indices)}
                    new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
                    atoms.info["cluster_partition"] = new_target_partition
                    atoms.info["cluster_center"] = cluster_center.tolist()

                elif truncation_mode == "local-convention" and cluster_radius is not None:
                    if _global_config.get("cluster_with_pseudonodes", False):
                        max_radius = cluster_radius + _global_config.get("pseudo_radius", 6.0)
                    else:
                        max_radius = cluster_radius

                    vol = atoms.get_volume()
                    a1, a2, a3 = atoms.cell
                    
                    rep_a1 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a2, a3)) / vol))
                    rep_a2 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a3, a1)) / vol))
                    rep_a3 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a1, a2)) / vol))
                    
                    M1, M2, M3 = 2 * rep_a1 + 1, 2 * rep_a2 + 1, 2 * rep_a3 + 1
                    
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

                    supercell = atoms * [M1, M2, M3]
                    supercell.info = atoms.info.copy()
                    supercell.pbc = [False, False, False]
                    
                    translation = rep_a1 * a1 + rep_a2 * a2 + rep_a3 * a3
                    new_cluster_center = cluster_center + translation
                    supercell.info["cluster_center"] = new_cluster_center.tolist()
                    
                    distances = np.linalg.norm(supercell.get_positions() - new_cluster_center, axis=1)
                    
                    target_indices = np.where(distances <= cluster_radius)[0]
                    retained_indices = np.where(distances <= max_radius)[0]
                    
                    if len(retained_indices) == 0:
                        logger.warning(f"Skipping {base_name}: retained_indices is empty in local-convention mode.")
                        continue
                    
                    target_partition_np = target_indices
                    target_weight_full = np.bincount(target_indices, minlength=len(supercell))
                    supercell.arrays["atom_weight"] = target_weight_full
                        
                    old_to_new = {old: new for new, old in enumerate(retained_indices)}
                    new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
                    supercell.info["cluster_partition"] = new_target_partition
                    
                    atoms = supercell[retained_indices]
                    
                    if has_forces and "forces" in atoms.arrays:
                        atoms.info["forces"] = atoms.arrays["forces"].tolist()
                        del atoms.arrays["forces"]
                        
                    if "atom_weight" in atoms.arrays:
                        atoms.info["atom_weight"] = atoms.arrays["atom_weight"].tolist()
                        del atoms.arrays["atom_weight"]
                
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

                db.write(atoms, data=atoms.info)
                natoms.append(len(atoms))
                successful.append(f"{os.path.basename(traj_file)}")
                
            except Exception as e:
                failed.append(f"{os.path.basename(traj_file)}: Unexpected error: {e!s}")

    return db_file, natoms, successful, failed


def launch_processing(
    traj_files: list[str],
    output_dir: str | Path,
    num_workers: int,
    mapping: dict,
    config: dict,
) -> None:
    """
    Launch parallel processing of traj files into ASE database.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not traj_files:
        logger.warning("No files to process!")
        return

    chunked_files = np.array_split(traj_files, num_workers)
    db_files = [
        output_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)
    ]

    mp_args = []
    for i in range(num_workers):
        if len(chunked_files[i]) > 0:
            mp_args.append((db_files[i], chunked_files[i], i))

    outputs = []
    if mp_args:
        with mp.Pool(
            num_workers,
            initializer=_init_processing_pool,
            initargs=(mapping, config),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    process_file_list, mp_args
                ),
                total=num_workers,
                desc="Processing workers",
            ):
                outputs.append(result)

    total_natoms = []
    total_successful = 0
    total_failed = 0

    outputs.sort(key=lambda x: str(x[0]))

    for output in outputs:
        db_file, natoms, successful, failed = output
        total_natoms.extend(natoms)
        total_successful += len(successful)
        total_failed += len(failed)

        log_file = db_file.with_suffix(".log")
        failed_file = db_file.with_suffix(".failed")

        with open(log_file, "w") as log:
            log.write("\n".join(successful))

        if failed:
            with open(failed_file, "w") as failed_log:
                failed_log.write("\n".join(failed))

    np.savez_compressed(output_dir / "metadata.npz", natoms=total_natoms)

    logger.info("Processing complete!")
    logger.info(f"Total structures processed: {total_successful}")
    logger.info(f"Total structures failed: {total_failed}")
    logger.info(f"Output saved to: {output_dir}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert .traj dataset to ASE LMDB format with an optional cluster center offset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing .traj files and id_prop_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output base directory (will create train/val/test subdirectories)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing files",
    )
    parser.add_argument(
        "--compute-normalizer",
        action="store_true",
        help="Compute adsorption_energy_rms from training data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting and offsetting",
    )
    parser.add_argument(
        "--truncation-mode",
        type=str,
        default="none",
        choices=["none", "local", "local-convention"],
        help="Physical graph truncation mode: none, local, or local-convention.",
    )
    parser.add_argument(
        "--cluster-radius",
        type=float,
        default=None,
        help="Radius (in Angstroms) around the cluster center to keep.",
    )
    parser.add_argument(
        "--pseudo-radius",
        type=float,
        default=6.0,
        help="Cutoff radius to buffer pseudonodes if pseudonodes are active.",
    )
    parser.add_argument(
        "--with-pseudonodes",
        action="store_true",
        default=False,
        help="Include buffer atoms outside the cluster radius (default: False).",
    )
    # --- NEW ARGUMENTS ---
    parser.add_argument(
        "--center-offset",
        type=float,
        default=0.0,
        help="Offset distance (in Angstroms) to randomly shift the cluster center for sensitivity analysis.",
    )
    parser.add_argument(
        "--offset-mode",
        type=str,
        default="xyz",
        choices=["xyz", "xy"],
        help="Shift the center randomly in 3D ('xyz') or in the 2D plane ('xy'). Use 'xy' for surface impurities.",
    )
    # ---------------------

    return parser


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    csv_file = input_dir / "id_prop_index.csv"
    if not csv_file.exists():
        logger.error(f"Cannot find CSV file: {csv_file}")
        return

    mapping = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["id"]] = {
                "adsorption_energy": float(row["adsorption_energy"]),
                "cidxs": int(row["cidxs"]) if row["cidxs"] else -1
            }
    
    logger.info(f"Loaded {len(mapping)} records from {csv_file}")

    traj_files = glob.glob(str(input_dir / "*.traj"))
    logger.info(f"Found {len(traj_files)} .traj files in {input_dir}")
    
    valid_traj_files = []
    for f in traj_files:
        base_name = os.path.basename(f).replace(".traj", "")
        if base_name in mapping:
            valid_traj_files.append(f)
            
    logger.info(f"Filtered to {len(valid_traj_files)} files that have entries in the CSV")

    valid_traj_files.sort()

    random.seed(args.seed)
    random.shuffle(valid_traj_files)
    
    n_total = len(valid_traj_files)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_files = valid_traj_files[:n_train]
    val_files = valid_traj_files[n_train:n_train+n_val]
    test_files = valid_traj_files[n_train+n_val:]
    
    logger.info(f"Splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    output_dir = Path(args.output_dir)
    
    config = {
        "truncation_mode": args.truncation_mode,
        "cluster_radius": args.cluster_radius,
        "pseudo_radius": args.pseudo_radius,
        "cluster_with_pseudonodes": args.with_pseudonodes,
        "center_offset": args.center_offset,
        "offset_mode": args.offset_mode,
        "seed": args.seed,
    }

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if files:
            split_dir = output_dir / split_name
            logger.info(f"Processing {split_name} dataset...")
            launch_processing(
                traj_files=files,
                output_dir=split_dir,
                num_workers=args.num_workers,
                mapping=mapping,
                config=config,
            )

    if args.compute_normalizer:
        train_dir = output_dir / "train"
        if train_dir.exists():
            logger.info("Computing normalizer (adsorption_energy_rms) for train set...")
            try:
                energy_rms = compute_adsorption_energy_rms(train_dir)
                
                normalizer_stats = {
                    "adsorption_energy_rms": energy_rms,
                }

                stats_file = train_dir / "normalizer_stats.json"
                with open(stats_file, "w") as f:
                    json.dump(normalizer_stats, f, indent=2)

                logger.info(f"Normalizer stats saved to {stats_file}")
                logger.info(f"Adsorption Energy RMS: {energy_rms}")
            except Exception as e:
                logger.error(f"Failed to compute normalizer: {e!s}", exc_info=True)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
