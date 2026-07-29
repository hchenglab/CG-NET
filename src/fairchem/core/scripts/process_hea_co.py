import os
import argparse
import csv
import glob
import numpy as np
from ase.io import read, write

def main():
    parser = argparse.ArgumentParser(description="Process hea_co and hea_surface into a fairchem-compatible traj dataset.")
    parser.add_argument('--input-dir', type=str, required=True, 
                        help="Input directory containing hea_co and hea_surface subdirectories")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory to save the processed .traj files and id_prop_index.csv")
    args = parser.parse_args()

    co_dir = os.path.join(args.input_dir, 'hea_co')
    surface_dir = os.path.join(args.input_dir, 'hea_surface')

    if not os.path.exists(co_dir):
        raise FileNotFoundError(f"Cannot find hea_co directory: {co_dir}")
    if not os.path.exists(surface_dir):
        raise FileNotFoundError(f"Cannot find hea_surface directory: {surface_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Reference energies (from previous pure_co script convention)
    Ecog = -14.327528
    dE_sol_co = -0.1

    csv_rows = []
    
    # Iterate over all .traj files in hea_co
    co_traj_files = glob.glob(os.path.join(co_dir, '*.traj'))
    
    success_count = 0
    for co_file in co_traj_files:
        basename = os.path.basename(co_file)
        sys_id = os.path.splitext(basename)[0]  # e.g., 's0'
        
        surface_file = os.path.join(surface_dir, basename)
        if not os.path.exists(surface_file):
            print(f"Warning: Corresponding surface file not found for {sys_id}, skipping.")
            continue
            
        try:
            # Read atoms
            atoms = read(co_file)
            surface = read(surface_file)
            
            # Extract energies
            E_sys = atoms.get_potential_energy()
            E_surface = surface.get_potential_energy()
            
            # Calculate Adsorption Energy
            E_ads = E_sys + dE_sol_co - E_surface - Ecog
            
            # Fix tags
            symbols = np.array(atoms.get_chemical_symbols())
            tags = np.ones(len(atoms), dtype=int)
            # Adsorbate gets tag=2
            ads_mask = (symbols == 'C') | (symbols == 'O')
            tags[ads_mask] = 2
            atoms.set_tags(tags)
            
            # Save processed traj
            out_traj = os.path.join(args.output_dir, basename)
            write(out_traj, atoms)
            
            csv_rows.append({
                'id': sys_id,
                'adsorption_energy': E_ads,
                'cidxs': -1 # fallback
            })
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {sys_id}: {e}")

    # Write CSV
    csv_file = os.path.join(args.output_dir, 'id_prop_index.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'adsorption_energy', 'cidxs'])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"\nProcessing complete!")
    print(f"Successfully processed {success_count} structures.")
    print(f"Output saved to {args.output_dir}")

if __name__ == '__main__':
    main()
