# Datasets

This directory contains information about the datasets used for CG-NET training and evaluation.

**Note: Due to the large size of the datasets, they have been uploaded to Figshare for easy access and sharing.**

## Available Datasets

The following datasets are available for download from Figshare:

- **hea.tar.gz**: High-entropy alloy (HEA) dataset
- **imp2d.tar.gz**: Two-dimensional impurity (2D-impurity) dataset - [Nature Materials](https://www.nature.com/articles/s41699-023-00380-6)
- **oc20.tar.gz**: Open Catalyst 2020 (OC20) dataset - [ACS Catalysis](https://pubs.acs.org/doi/10.1021/acscatal.0c04525)
- **odac23.tar.gz**: Open DAC 2023 (ODAC23) dataset - [ACS Central Science](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629)

## Download Instructions

1. Download the datasets from Figshare: [https://figshare.com/articles/dataset/Datasets_used_for_CG-NET_training_and_evaluation_/29413064]
2. Extract the datasets using: `tar -zxvf *.tar.gz`
3. Place the extracted directories in this `datasets/` folder

The final structure should look like:
```
datasets/
├── README.md
├── hea/
├── imp2d/
├── oc20/
└── odac23/
```

## Data Format

Each dataset directory should contain:
- Structure files in ASE-compatible trajectory formats (`.traj`)
- A CSV file named `id_prop_index.csv` with columns:
  - `id`: Identifier matching the structure file name
  - `label`: Target property value
  - `cidxs`: Atom indices identifying the cluster center (for multiple centers, use `cidx_1, cidx_2, ...`)

Please refer to the main README.md for detailed dataset descriptions and experimental setup.
