import os
from ase.io import read
import numpy as np

def process_id_prop(id: str, data_path: str = "../data") -> str:
    """
    Process a single structure file to find the closest atom index for tag=2 atoms.

    Args:
        id (str): The ID of the structure to process.
        data_path (str): Path to the directory containing the .traj files.

    Returns:
        str: The closest atom index for tag=2 atoms.
    """
    # Read the structure file
    structure = read(os.path.join(data_path, f"{id}.traj"))

    # Find indices of atoms with tag=2 and tag!=2
    tag_2_indices = [i for i, atom in enumerate(structure) if atom.tag == 2]
    non_tag_2_indices = [i for i, atom in enumerate(structure) if atom.tag != 2]

    # Initialize variables to track the minimum distance and corresponding index
    min_distance = float('inf')
    closest_tag_2_index = None

    # Calculate distances between tag=2 atoms and tag!=2 atoms
    for tag_2_index in tag_2_indices:
        for non_tag_2_index in non_tag_2_indices:
            distance = structure.get_distance(tag_2_index, non_tag_2_index)
            if distance < min_distance:
                min_distance = distance
                closest_tag_2_index = tag_2_index

    # Return the closest tag=2 index as a string
    return str(closest_tag_2_index)