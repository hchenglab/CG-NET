# Project Title: Process ID Properties

## Overview
This project is designed to process atomic data from a CSV file, specifically `id_prop_index.csv`. It reads the corresponding trajectory files for each atomic ID, calculates the nearest distances between specific types of atoms, and generates new information for further analysis.

## Purpose
The main goal of this project is to enhance the existing atomic data by adding new `cidxs` information based on the proximity of atoms with specific tags. This will facilitate more detailed analyses in subsequent research.

## File Structure
```
process-id-prop
├── src
│   ├── main.py
│   ├── utils
│   │   └── file_processor.py
├── data
│   └── id_prop_index.csv
├── requirements.txt
└── README.md
```

## Installation
To set up the project, ensure you have Python installed on your machine. Then, create a virtual environment and install the required packages using the following commands:

```bash
pip install -r requirements.txt
```

## Usage
1. Place your `id_prop_index.csv` file in the `data` directory.
2. Ensure that the corresponding `.traj` files are accessible in the same directory as the CSV file.
3. Run the main script to process the data:

```bash
python src/main.py
```

This will read the `id_prop_index.csv`, process the data to find the nearest distances, and output the updated information back to a new CSV file.

## Requirements
The project requires the following Python libraries:
- ase
- pandas

Make sure to install these dependencies before running the project.

## Contribution
Contributions to improve the functionality or efficiency of the project are welcome. Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.