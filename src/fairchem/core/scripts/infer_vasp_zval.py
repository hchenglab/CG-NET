"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Infer the VASP POTCAR valence electron counts (ZVAL) by solving a system
of linear equations derived from the total electrons reported in ACF.dat
files and the stoichiometry of the systems.
"""

import argparse
import glob
import os
import lzma
import numpy as np
from ase.io import read
from tqdm import tqdm
import tempfile

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Infer VASP ZVALs from ACF.dat and extxyz.xz files")
    parser.add_argument("--input-dirs", type=str, nargs="+", required=True, help="Path(s) to dataset directories (e.g. ads_O/0)")
    parser.add_argument("--bader-dir", type=str, required=True, help="Path to bader data directory containing randomXXXXX subdirs")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum number of files to parse")
    return parser

def main():
    args = get_parser().parse_args()

    extxyz_files = []
    for d in args.input_dirs:
        extxyz_files.extend(glob.glob(os.path.join(d, "**/*.extxyz.xz"), recursive=True))

    extxyz_files = extxyz_files[:args.max_samples]
    if not extxyz_files:
        print("No files found!")
        return

    X = []
    y = []
    elements_list = []

    print(f"Parsing up to {len(extxyz_files)} files...")
    
    for f in tqdm(extxyz_files):
        sysid = os.path.basename(f).split('.')[0]
        acf_path = os.path.join(args.bader_dir, sysid, "ACF.dat")
        if not os.path.exists(acf_path):
            continue
        
        # Read the last frame of the trajectory to get elements
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                extxyz_file = os.path.join(tmpdir, "structures.extxyz")
                with open(f, "rb") as fin:
                    extxyz_content = lzma.decompress(fin.read())
                    with open(extxyz_file, "wb") as out:
                        out.write(extxyz_content)
                atoms = read(extxyz_file, index='-1')
            except Exception:
                continue
            
        symbols = atoms.get_chemical_symbols()
        
        total_elec = 0.0
        try:
            with open(acf_path) as acf:
                lines = acf.readlines()
                for l in reversed(lines):
                    if "NUMBER OF ELECTRONS:" in l:
                        total_elec = float(l.split()[-1])
                        break
        except Exception:
            continue
                    
        counts = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
            if s not in elements_list:
                elements_list.append(s)
                
        row = [counts.get(el, 0) for el in elements_list]
        X.append(row)
        y.append(total_elec)

    # Pad X to uniform size since elements_list grows dynamically
    X_matrix = np.zeros((len(X), len(elements_list)))
    for i, row in enumerate(X):
        X_matrix[i, :len(row)] = row
        
    y_array = np.array(y)

    print(f"Successfully parsed {len(y_array)} systems containing {len(elements_list)} unique elements.")
    print("Solving linear system...")
    
    Z, residuals, rank, s = np.linalg.lstsq(X_matrix, y_array, rcond=None)

    print("\nInferred VASP ZVALs (Valence Electrons):")
    print("OC20_ZVALS = {")
    for el, z in zip(elements_list, Z):
        print(f"    '{el}': {int(round(z))},")
    print("}")

    Z_rounded = np.array([int(round(z)) for z in Z])
    y_pred = X_matrix @ Z_rounded
    errors = np.abs(y_array - y_pred)
    print(f"\nPrediction Error (using rounded ZVALs):")
    print(f"Mean Absolute Error: {np.mean(errors):.4f} electrons")
    print(f"Max Absolute Error: {np.max(errors):.4f} electrons")

if __name__ == "__main__":
    main()
