"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Converts OC20 S2EF dataset (compressed .extxyz.xz and .txt.xz files) into
ASE LMDB database format suitable for UMA model training and evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import lzma
import multiprocessing as mp
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from ase.db import connect
from ase.io import read
from tqdm import tqdm

from fairchem.core.datasets import AseDBDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for multiprocessing
_global_dataset = None
_global_mapping = None


def _init_processing_pool(mapping_file: str | None) -> None:
    """
    Initialize global mapping dictionary for worker processes.
    """
    global _global_mapping
    if mapping_file and os.path.exists(mapping_file):
        import pickle
        with open(mapping_file, "rb") as f:
            _global_mapping = pickle.load(f)


def extract_energy_and_forces(idx: int, dataset: AseDBDataset) -> tuple:
    """
    Extract energy and forces from an ASE atoms object at a given index.

    Args:
        idx: Index in the dataset
        dataset: AseDBDataset instance

    Returns:
        Tuple of (atomic_numbers, energy, forces)
    """
    atoms = dataset.get_atoms(idx)
    atomic_numbers = atoms.get_atomic_numbers()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    n_atoms = len(atoms)

    fixed_idx = np.zeros(n_atoms)
    if hasattr(atoms, "constraints"):
        from ase.constraints import FixAtoms

        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed_idx[constraint.index] = 1

    mask = fixed_idx == 0
    forces = forces[mask]

    return atomic_numbers, energy, forces


def _extract_for_pool(idx: int) -> tuple:
    """
    Wrapper for pool.imap to extract energy and forces.
    Uses the global dataset variable.
    """
    return extract_energy_and_forces(idx, _global_dataset)


def _init_pool(db_pattern: str) -> None:
    """
    Initialize global dataset for worker processes.
    
    Args:
        db_pattern: Glob pattern for the LMDB database files.
    """
    global _global_dataset
    _global_dataset = AseDBDataset({"src": db_pattern})


def compute_lin_ref(atomic_numbers: list, energies: list) -> list:
    """
    Compute linear reference coefficients given atomic numbers and energies.

    Args:
        atomic_numbers: List of atomic number arrays
        energies: List of energy values

    Returns:
        List of linear reference coefficients
    """
    features = [
        np.bincount(x, minlength=100).astype(int) for x in atomic_numbers
    ]

    X = np.vstack(features)
    y = np.array(energies)

    coeff = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.isfinite(
        coeff
    ).all(), (
        "Found some non-finite values while computing element references, "
        "please check/sanitize your inputs before proceeding"
    )
    return coeff.tolist()


def compute_normalizer_and_linear_reference(
    train_path: str | Path,
    num_workers: int,
    sample_size: int = 100000,
) -> tuple[float, list]:
    """
    Compute the normalizer value and linear reference coefficients from
    training LMDB dataset.

    Args:
        train_path: Path to the training LMDB database directory
        num_workers: Number of workers for multiprocessing
        sample_size: Maximum number of samples to use (default: 100000)

    Returns:
        Tuple of (force_rms, linear_coeff)
    """
    # Build glob pattern to find all LMDB databases
    train_path = Path(train_path)
    db_pattern = str(train_path / "data.*.aselmdb")

    # Check how many entries are in the dataset to correctly sample
    # This is a workaround for the fact that AseDBDataset is not picklable
    temp_dataset = AseDBDataset({"src": db_pattern})
    dataset_size = len(temp_dataset)
    del temp_dataset # free up memory

    sample_count = min(sample_size, dataset_size)
    sample_indices = random.sample(range(dataset_size), sample_count)

    # Use initializer to pass db_pattern to worker processes
    with mp.Pool(num_workers, initializer=_init_pool, initargs=(db_pattern,)) as pool:
        outputs = list(
            tqdm(
                pool.imap(_extract_for_pool, sample_indices),
                total=len(sample_indices),
                desc="Computing normalizer values.",
            )
        )

    atomic_numbers = [x[0] for x in outputs]
    energies = [x[1] for x in outputs]
    forces = np.array([force for x in outputs for force in x[2]])
    force_rms = np.sqrt(np.mean(np.square(forces)))

    assert np.isfinite(
        force_rms
    ).all(), (
        "We found non-finite values in the forces, "
        "please check your input data!"
    )
    coeff = compute_lin_ref(atomic_numbers, energies)

    return float(force_rms), coeff


def parse_metadata(txt_file: str) -> list[dict]:
    """
    Parse metadata from decompressed .txt file.

    Each line contains: system_id,frame_number,reference_energy
    (no header line)

    Args:
        txt_file: Path to decompressed .txt file

    Returns:
        List of dicts with keys: system_id, frame_number, reference_energy
    """
    metadata_list = []
    
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(
                    f"Expected 3 columns in metadata line, got {len(parts)}: {line}"
                )

            system_id, frame_str, energy_str = parts
            
            # Extract frame number from "frame123" format
            frame_str = frame_str.strip()
            if frame_str.lower().startswith("frame"):
                try:
                    frame_number = int(frame_str[5:])
                except ValueError:
                    frame_number = int(frame_str)
            else:
                frame_number = int(frame_str)

            try:
                ref_energy = float(energy_str)
            except ValueError:
                raise ValueError(
                    f"Could not parse energy as float: {energy_str}"
                )

            metadata_list.append(
                {
                    "system_id": system_id.strip(),
                    "frame_number": frame_number,
                    "reference_energy": ref_energy,
                }
            )

    return metadata_list


def process_file_pair(mp_arg: tuple) -> tuple[str, list, list, list]:
    """
    Process a pair of .extxyz and .txt files (compressed) and write to ASE DB.

    Args:
        mp_arg: Tuple of (db_file, extxyz_xz_file, txt_xz_file, worker_id)

    Returns:
        Tuple of (db_file, natoms_list, successful_list, failed_list)
    """
    db_file, extxyz_xz_file, txt_xz_file, worker_id = mp_arg

    successful = []
    failed = []
    natoms = []

    # Create temporary directory for decompressed files
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Decompress metadata file
            txt_file = os.path.join(tmpdir, "metadata.txt")
            with open(txt_xz_file, "rb") as f:
                txt_content = lzma.decompress(f.read())
                with open(txt_file, "wb") as out:
                    out.write(txt_content)
            
            # Parse metadata from file
            metadata_list = parse_metadata(txt_file)

            # Decompress structures file
            extxyz_file = os.path.join(tmpdir, "structures.extxyz")
            with open(extxyz_xz_file, "rb") as f:
                extxyz_content = lzma.decompress(f.read())
                with open(extxyz_file, "wb") as out:
                    out.write(extxyz_content)
            
            # Parse structures from extxyz file
            atoms_list = read(extxyz_file, index=":")

            # Validate matching counts
            if len(atoms_list) != len(metadata_list):
                failed.append(
                    f"{os.path.basename(extxyz_xz_file)}: "
                    f"Mismatch between structures ({len(atoms_list)}) "
                    f"and metadata ({len(metadata_list)})"
                )
                return db_file, natoms, successful, failed

            # Calculate adsorption energies and add to metadata
            for idx, metadata in enumerate(metadata_list):
                atoms = atoms_list[idx]
                reference_energy = metadata["reference_energy"]
                adsorption_energy = atoms.get_potential_energy() - reference_energy
                metadata["adsorption_energy"] = adsorption_energy

                system_id = metadata["system_id"]
                if _global_mapping is None or system_id not in _global_mapping:
                    raise ValueError(f"Mapping not available for system_id: {system_id}")

                # Calculate cluster center from actual adsorbate positions in the current frame
                tags = atoms.get_tags()
                ads_mask = (tags == 2)
                if ads_mask.any():
                    cluster_center = atoms.get_positions()[ads_mask].mean(axis=0).tolist()
                else:
                    # Fallback to initial adsorption_site if tags are missing or no adsorbate
                    adsorption_site = _global_mapping[system_id].get("adsorption_site")
                    if not adsorption_site:
                        raise ValueError(f"adsorption_site missing or empty for system_id: {system_id}")
                    cluster_center = np.mean(adsorption_site, axis=0).tolist()

                metadata["cluster_center"] = cluster_center

            # Write to database
            with connect(str(db_file)) as db:
                for i, (atoms, metadata) in enumerate(zip(atoms_list, metadata_list)):
                    try:
                        # Validate required fields
                        assert (
                            atoms.calc is not None
                        ), "No calculator attached to atoms object."
                        assert "energy" in atoms.calc.results, "Missing energy result"
                        assert "forces" in atoms.calc.results, "Missing forces result"

                        # Add metadata to atoms.info
                        if atoms.info is None:
                            atoms.info = {}
                        atoms.info.update(metadata)

                        # Write to database
                        db.write(atoms, data=atoms.info)
                        natoms.append(len(atoms))
                        successful.append(
                            f"{os.path.basename(extxyz_xz_file)},{i}"
                        )
                    except AssertionError as e:
                        failed.append(
                            f"{os.path.basename(extxyz_xz_file)},{i}: {e!s}"
                        )
                    except Exception as e:
                        failed.append(
                            f"{os.path.basename(extxyz_xz_file)},{i}: "
                            f"Unexpected error: {e!s}"
                        )

        except Exception as e:
            failed.append(
                f"{os.path.basename(extxyz_xz_file)}: "
                f"File processing error: {e!s}"
            )

    return db_file, natoms, successful, failed


def _process_worker_chunk(
    mp_arg: tuple,
) -> tuple[str | Path, list, list, list]:
    """
    Process a chunk of file pairs assigned to a worker.

    Args:
        mp_arg: Tuple of (db_file, file_pairs_list, worker_id)

    Returns:
        Tuple of (db_file, all_natoms, all_successful, all_failed)
    """
    db_file, file_pairs, worker_id = mp_arg

    all_natoms = []
    all_successful = []
    all_failed = []

    for extxyz_file, txt_file in tqdm(
        file_pairs, position=worker_id, desc=f"Worker {worker_id}"
    ):
        _, natoms, successful, failed = process_file_pair(
            (db_file, extxyz_file, txt_file, worker_id)
        )
        all_natoms.extend(natoms)
        all_successful.extend(successful)
        all_failed.extend(failed)

    return db_file, all_natoms, all_successful, all_failed



def prepare_file_pairs(
    input_dirs: list[str],
) -> list[tuple[str, str]]:
    """
    Prepare list of (extxyz_xz, txt_xz) file pairs from input directories.

    Args:
        input_dirs: List of directories containing compressed files

    Returns:
        List of (extxyz_xz_path, txt_xz_path) tuples
    """
    file_pairs = []

    for input_dir in input_dirs:
        extxyz_files = sorted(
            glob.glob(os.path.join(input_dir, "*.extxyz.xz"))
        )

        for extxyz_file in extxyz_files:
            # Find corresponding txt file
            base_name = os.path.basename(extxyz_file).replace(
                ".extxyz.xz", ""
            )
            txt_file = os.path.join(input_dir, f"{base_name}.txt.xz")

            if not os.path.exists(txt_file):
                logger.warning(
                    f"Missing metadata file for {extxyz_file}: {txt_file}"
                )
                continue

            file_pairs.append((extxyz_file, txt_file))

    return file_pairs


def launch_processing(
    input_dirs: list[str],
    output_dir: str | Path,
    num_workers: int,
    mapping_file: str | None = None,
) -> None:
    """
    Launch parallel processing of file pairs into ASE database.

    Args:
        input_dirs: List of input directories containing compressed files
        output_dir: Output directory for ASE LMDB database
        num_workers: Number of parallel workers
        mapping_file: Path to mapping pkl file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare file pairs
    file_pairs = prepare_file_pairs(input_dirs)
    logger.info(f"Found {len(file_pairs)} file pairs to process")

    if not file_pairs:
        logger.warning("No file pairs found in input directories!")
        return

    # Split work across workers
    chunked_pairs = np.array_split(file_pairs, num_workers) # type: ignore
    db_files = [
        output_dir / f"data.{i:04d}.aselmdb" for i in range(num_workers)
    ]

    # Prepare multiprocessing arguments
    mp_args = []
    for i in range(num_workers):
        if len(chunked_pairs[i]) > 0:
            mp_args.append((db_files[i], chunked_pairs[i], i))

    # Process in parallel
    outputs = []
    if mp_args:
        with mp.Pool(
            num_workers,
            initializer=_init_processing_pool,
            initargs=(mapping_file,),
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(
                    _process_worker_chunk, mp_args
                ),
                total=num_workers,
                desc="Processing workers",
            ):
                outputs.append(result)

    # Log results
    total_natoms = []
    total_successful = 0
    total_failed = 0

    # SORT outputs by db_file to ensure natoms align perfectly with the alphabetically
    # sorted dataset files when loaded by AseDBDataset later. Because pool.imap_unordered
    # returns chunks in completion order, failure to sort causes catastrophic misalignment.
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

    # Save metadata
    np.savez_compressed(output_dir / "metadata.npz", natoms=total_natoms)

    logger.info(f"Processing complete!")
    logger.info(f"Total structures processed: {total_successful}")
    logger.info(f"Total structures failed: {total_failed}")
    logger.info(f"Output saved to: {output_dir}")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert OC20 S2EF dataset to ASE LMDB format"
    )
    parser.add_argument(
        "--train-input-dirs",
        type=str,
        nargs="+",
        help="Path(s) to training dataset directories with compressed files",
    )
    parser.add_argument(
        "--val-input-dirs",
        type=str,
        nargs="+",
        help="Path(s) to validation dataset directories with compressed files",
    )
    parser.add_argument(
        "--output-train-dir",
        type=str,
        help="Output directory for training ASE LMDB database",
    )
    parser.add_argument(
        "--output-val-dir",
        type=str,
        help="Output directory for validation ASE LMDB database",
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
        help="Compute force_rms and linear reference coefficients from training data",
    )
    parser.add_argument(
        "--normalizer-sample-size",
        type=int,
        default=100000,
        help="Number of samples to use for computing normalizer (default: 100000)",
    )
    parser.add_argument(
        "--oc20-mapping-file",
        type=str,
        default="/Users/chenghao/research/datasets/oc20/oc20_data_mapping.pkl",
        help="Path to oc20_data_mapping.pkl to extract adsorption sites",
    )
    return parser


def main(args: argparse.Namespace) -> None:
    """
    Main entry point.
    """
    normalizer_stats = None

    # Process training data if specified
    if args.train_input_dirs and args.output_train_dir:
        logger.info("Processing training dataset...")
        launch_processing(
            args.train_input_dirs,
            args.output_train_dir,
            args.num_workers,
            args.oc20_mapping_file,
        )

        # Compute normalizer if requested
        if args.compute_normalizer:
            logger.info("Computing normalizer and linear reference...")
            try:
                force_rms, linear_coeff = (
                    compute_normalizer_and_linear_reference(
                        args.output_train_dir,
                        args.num_workers,
                        args.normalizer_sample_size,
                    )
                )
                normalizer_stats = {
                    "force_rms": force_rms,
                    "linear_ref": linear_coeff,
                }

                # Save to JSON file
                stats_file = Path(args.output_train_dir) / "normalizer_stats.json"
                with open(stats_file, "w") as f:
                    json.dump(normalizer_stats, f, indent=2)

                logger.info(f"Normalizer stats saved to {stats_file}")
                logger.info(f"Force RMS: {force_rms}")
                logger.info(
                    f"Linear reference coefficients (100 elements): "
                    f"{linear_coeff[:10]} ... (showing first 10)"
                )
            except Exception as e:
                logger.error(
                    f"Failed to compute normalizer: {e!s}", exc_info=True
                )

    # Process validation data if specified
    if args.val_input_dirs and args.output_val_dir:
        logger.info("Processing validation dataset...")
        launch_processing(
            args.val_input_dirs,
            args.output_val_dir,
            args.num_workers,
            args.oc20_mapping_file,
        )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Validate arguments
    if not args.train_input_dirs and not args.val_input_dirs:
        parser.error(
            "Please specify at least one of --train-input-dirs or "
            "--val-input-dirs"
        )

    if args.train_input_dirs and not args.output_train_dir:
        parser.error(
            "Must specify --output-train-dir when using --train-input-dirs"
        )

    if args.val_input_dirs and not args.output_val_dir:
        parser.error(
            "Must specify --output-val-dir when using --val-input-dirs"
        )

    main(args)
