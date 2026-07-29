"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Converts OC20 IS2RES dataset and Bader Charge data into
ASE LMDB database format suitable for UMA model training and evaluation.
Only the relaxed frame (frame -1) is saved with the Bader net charges.
"""

from __future__ import annotations

import argparse
import glob
import logging
import lzma
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm
import random
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Exact VASP 5.4 PBE ZVALs inferred directly from OC20 ACF.dat
OC20_ZVALS = {
    'Co': 9, 'Hf': 4, 'O': 6, 'Bi': 5, 'Al': 3, 'Zr': 12, 'Sb': 5, 'Rh': 9, 'P': 5, 'Ca': 8,
    'In': 3, 'Ru': 8, 'Sc': 11, 'Ti': 4, 'Si': 4, 'Ni': 10, 'Sr': 10, 'S': 6, 'K': 7, 'Au': 11,
    'Y': 11, 'N': 5, 'Tl': 3, 'Pb': 4, 'Zn': 12, 'Cl': 7, 'As': 5, 'Ga': 3, 'Nb': 11, 'Ir': 9,
    'V': 5, 'Rb': 7, 'Mn': 7, 'Te': 6, 'Mo': 6, 'W': 6, 'Se': 6, 'Cu': 11, 'Na': 1, 'H': 1,
    'Pt': 10, 'Os': 8, 'Ta': 5, 'Pd': 10, 'Hg': 12, 'C': 4, 'Cs': 9, 'Fe': 8, 'Ag': 11, 'Ge': 4,
    'Cr': 6, 'Sn': 4, 'Cd': 12, 'Tc': 7, 'Re': 7,
}

_global_metadata = {}
_global_config = {}

def parse_metadata(system_txt: str) -> dict[str, float]:
    metadata = {}
    with open(system_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) == 2:
                system_id, energy_str = parts
                metadata[system_id.strip()] = float(energy_str)
    return metadata

def _init_processing_pool(system_txt_path: str, config: dict) -> None:
    global _global_metadata, _global_config
    if system_txt_path and os.path.exists(system_txt_path):
        _global_metadata = parse_metadata(system_txt_path)
    _global_config = config

def parse_bader_acf(acf_path: str) -> tuple[np.ndarray, float]:
    """Parse ACF.dat and return an array of CHARGE values and total electrons."""
    charges = []
    total_electrons = 0.0
    with open(acf_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                charges.append(float(parts[4]))
            elif "NUMBER OF ELECTRONS:" in line:
                total_electrons = float(parts[-1])
    return np.array(charges), total_electrons

def process_file_pair(mp_arg: tuple) -> tuple[str, list, list, list]:
    db_file, extxyz_xz_file, bader_dir, worker_id = mp_arg
    successful = []
    failed = []
    natoms_list = []

    system_id = os.path.basename(extxyz_xz_file).split(".")[0]
    acf_path = os.path.join(bader_dir, system_id, "ACF.dat")

    if not os.path.exists(acf_path):
        failed.append(f"{system_id}: ACF.dat not found")
        return db_file, natoms_list, successful, failed

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Decompress structures file
            extxyz_file = os.path.join(tmpdir, "structures.extxyz")
            with open(extxyz_xz_file, "rb") as f:
                extxyz_content = lzma.decompress(f.read())
                with open(extxyz_file, "wb") as out:
                    out.write(extxyz_content)

            # Read the last frame of the uncompressed trajectory
            atoms = read(extxyz_file, index="-1")
            
        bader_charges_raw, acf_total_electrons = parse_bader_acf(acf_path)

        if len(atoms) != len(bader_charges_raw):
            failed.append(f"{system_id}: Atoms count mismatch ({len(atoms)} vs {len(bader_charges_raw)})")
            return db_file, natoms_list, successful, failed

        # Try to calculate net charges
        symbols = atoms.get_chemical_symbols()
        try:
            zvals = np.array([OC20_ZVALS[s] for s in symbols])
            expected_total = zvals.sum()
            bader_net_charges = zvals - bader_charges_raw
        except KeyError as e:
            failed.append(f"{system_id}: Missing element in OC20_ZVALS - {e!s}")
            return db_file, natoms_list, successful, failed

        # Temporarily place these per-atom properties into arrays 
        # so that ASE automatically handles tiling (for convention) and slicing (for local)
        atoms.arrays["bader_charge_raw"] = bader_charges_raw
        atoms.arrays["bader_charge"] = bader_net_charges

        if atoms.info is None:
            atoms.info = {}

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

        # Calculate cluster center from actual adsorbate positions
        tags = atoms.get_tags()
        ads_mask = (tags == 2)
        if ads_mask.any():
            cluster_center = atoms.get_positions()[ads_mask].mean(axis=0)
        else:
            cluster_center = atoms.get_positions().mean(axis=0)

        truncation_mode = _global_config.get("truncation_mode", "none")
        cluster_radius = _global_config.get("cluster_radius")

        if truncation_mode == "local" and cluster_radius is not None:
            from pymatgen.io.ase import AseAtomsAdaptor
            struct = AseAtomsAdaptor.get_structure(atoms)
            
            # Target atoms (within cluster_radius)
            target_sites = struct.get_sites_in_sphere(cluster_center.tolist(), cluster_radius)
            target_indices = [s.index for s in target_sites]
            target_weight_full = np.bincount(target_indices, minlength=len(atoms))
            target_partition_np = np.unique(target_indices)
            
            # All retained atoms (Target + Source buffer)
            if _global_config.get("cluster_with_pseudonodes", False):
                radius = _global_config.get("cutoff_radius", 6.0)
                all_sites = struct.get_sites_in_sphere(cluster_center.tolist(), cluster_radius + radius)
                retained_indices = np.unique([s.index for s in all_sites])
            else:
                retained_indices = target_partition_np
            
            # Safely slice the atoms object (handles arrays natively)
            atoms = atoms[retained_indices]
            
            atoms.info["atom_weight"] = target_weight_full[retained_indices].tolist()
            
            # Map old target indices to new relative indices
            old_to_new = {old: new for new, old in enumerate(retained_indices)}
            new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
            atoms.info["cluster_partition"] = new_target_partition
                
            atoms.info["cluster_center"] = cluster_center.tolist()

        elif truncation_mode == "local-convention" and cluster_radius is not None:
            # 1. Determine supercell size
            if _global_config.get("cluster_with_pseudonodes", False):
                max_radius = cluster_radius + _global_config.get("cutoff_radius", 6.0)
            else:
                max_radius = cluster_radius

            vol = atoms.get_volume()
            a1, a2, a3 = atoms.cell
            
            rep_a1 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a2, a3)) / vol))
            rep_a2 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a3, a1)) / vol))
            rep_a3 = int(np.ceil(max_radius * np.linalg.norm(np.cross(a1, a2)) / vol))
            
            M1, M2, M3 = 2 * rep_a1 + 1, 2 * rep_a2 + 1, 2 * rep_a3 + 1
            
            # 2. Build supercell
            supercell = atoms * [M1, M2, M3]
            supercell.info = atoms.info.copy()
            supercell.pbc = [False, False, False]
            
            # 3. Shift cluster center to the middle unit cell
            translation = rep_a1 * a1 + rep_a2 * a2 + rep_a3 * a3
            new_cluster_center = cluster_center + translation
            supercell.info["cluster_center"] = new_cluster_center.tolist()
            
            # 4. Get indices in sphere
            distances = np.linalg.norm(supercell.get_positions() - new_cluster_center, axis=1)
            
            target_indices = np.where(distances <= cluster_radius)[0]
            retained_indices = np.where(distances <= max_radius)[0]
            
            # 5. Apply mapping if saving cluster_partition and atom_weight
            target_partition_np = target_indices
            target_weight_full = np.bincount(target_indices, minlength=len(supercell))
            supercell.arrays["atom_weight"] = target_weight_full
                
            old_to_new = {old: new for new, old in enumerate(retained_indices)}
            new_target_partition = np.array([old_to_new[idx] for idx in target_partition_np])
            supercell.info["cluster_partition"] = new_target_partition
            
            # 6. Safely slice the atoms object
            atoms = supercell[retained_indices]
            
        else:
            atoms.info["cluster_center"] = cluster_center.tolist()

        # Add to info to persist
        if system_id in _global_metadata:
            atoms.info["reference_energy"] = _global_metadata[system_id]
        
        atoms.info["system_id"] = system_id
        
        # Post-process: move custom arrays to info as lists (to match OCP dataset conventions)
        for key in ["bader_charge_raw", "bader_charge", "atom_weight"]:
            if key in atoms.arrays:
                atoms.info[key] = atoms.arrays[key].tolist()
                del atoms.arrays[key]
                
        if has_forces and "forces" in atoms.arrays:
            atoms.info["forces"] = atoms.arrays["forces"].tolist()
            del atoms.arrays["forces"]

        with connect(str(db_file)) as db:
            db.write(atoms, data=atoms.info)
            
        natoms_list.append(len(atoms))
        successful.append(f"{system_id}")

    except Exception as e:
        failed.append(f"{system_id}: Unexpected error: {e!s}")

    return db_file, natoms_list, successful, failed

def _process_worker_chunk(mp_arg: tuple) -> tuple[str | Path, list, list, list]:
    db_file, chunked_files, bader_dir, worker_id = mp_arg
    all_natoms = []
    all_successful = []
    all_failed = []

    for extxyz_file in tqdm(chunked_files, position=worker_id, desc=f"Worker {worker_id}"):
        _, natoms, successful, failed = process_file_pair((db_file, extxyz_file, bader_dir, worker_id))
        all_natoms.extend(natoms)
        all_successful.extend(successful)
        all_failed.extend(failed)

    return db_file, all_natoms, all_successful, all_failed

def launch_processing(
    extxyz_files: list[str],
    bader_dir: str,
    system_txt_path: str,
    output_dir: str | Path,
    num_workers: int,
    config: dict,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(extxyz_files)} extxyz files into {output_dir}")

    if not extxyz_files:
        logger.warning("No files found!")
        return

    chunked_files = np.array_split(extxyz_files, num_workers) # type: ignore
    db_files = [output_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)]

    mp_args = []
    for i in range(num_workers):
        if len(chunked_files[i]) > 0:
            mp_args.append((db_files[i], chunked_files[i], bader_dir, i))

    outputs = []
    if mp_args:
        with mp.Pool(
            num_workers,
            initializer=_init_processing_pool,
            initargs=(system_txt_path, config),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(_process_worker_chunk, mp_args),
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

    logger.info(f"Processing complete!")
    logger.info(f"Total structures processed: {total_successful}")
    logger.info(f"Total structures failed: {total_failed}")
    logger.info(f"Output saved to: {output_dir}")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert OC20 Bader dataset to ASE LMDB format")
    parser.add_argument("--input-dirs", type=str, nargs="+", required=True, help="Path(s) to dataset directories (e.g. ads_O/0)")
    parser.add_argument("--bader-dir", type=str, required=True, help="Path to bader data directory containing randomXXXXX subdirs")
    parser.add_argument("--system-txt", type=str, required=True, help="Path to system.txt containing reference energies")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for ASE LMDB database")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    
    # Truncation arguments (same as create_traj_dataset.py)
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
        "--cutoff-radius",
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

    return parser

def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    
    extxyz_files = []
    for d in args.input_dirs:
        extxyz_files.extend(glob.glob(os.path.join(d, "**/*.extxyz.xz"), recursive=True))

    logger.info(f"Found {len(extxyz_files)} extxyz files total in {args.input_dirs}")
    
    if not extxyz_files:
        logger.error("No extxyz files found!")
        return
        
    random.seed(args.seed)
    random.shuffle(extxyz_files)
    
    n_total = len(extxyz_files)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_files = extxyz_files[:n_train]
    val_files = extxyz_files[n_train:n_train+n_val]
    test_files = extxyz_files[n_train+n_val:]
    
    logger.info(f"Splits: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    output_dir = Path(args.output_dir)
    
    config = {
        "truncation_mode": args.truncation_mode,
        "cluster_radius": args.cluster_radius,
        "cutoff_radius": args.cutoff_radius,
        "cluster_with_pseudonodes": args.with_pseudonodes,
    }

    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        if files:
            split_dir = output_dir / split_name
            logger.info(f"--- Processing {split_name} dataset ---")
            launch_processing(files, args.bader_dir, args.system_txt, split_dir, args.num_workers, config)

if __name__ == "__main__":
    main()
