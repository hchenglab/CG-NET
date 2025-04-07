# File: /process-id-prop/process-id-prop/src/main.py

import os
import pandas as pd
from utils.file_processor import process_id_prop

def main():
    # Define the path to the CSV file
    csv_file_path = os.path.join("..", "data", "id_prop_index.csv")
    
    # Read the CSV file
    data = pd.read_csv(csv_file_path)
    
    # Process the data to get new cidxs
    data['cidxs'] = data['id'].apply(lambda x: process_id_prop(x))
    
    # Write the updated data back to the CSV file
    data.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    main()