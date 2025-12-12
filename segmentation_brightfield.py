import os
import sys
# Import the class from your provided loader.py
from loader import DataLoader

def run_pipeline(target_folder_name):
    root_dir = os.path.join(os.getcwd(), "source")
    
    # Initialize the loader with the logic from loader.py
    data_loader = DataLoader(root_dir, target_folder_name)
    
    print(f"Processing folder: {data_loader.sample_folder}")

    # Use the loader to get valid file pairs (Brightfield, Fluorescence, XML)
    # This replaces the need to look for 'meta.json' manually
    valid_sets, errors = data_loader.get_file_pairs(data_loader.sample_folder)

    if not valid_sets:
        print("No valid data found.")
        if errors:
            print("Errors encountered:")
            for e in errors:
                print(f" - {e}")
        return

    print(f"Found {len(valid_sets)} valid image sets to process.")

    # Loop through the valid sets found by loader.py
    for item in valid_sets:
        sample_id = item['id']
        bf_path = item['bf']   # Path to _ch00.tif
        fl_path = item['fl']   # Path to _ch01.tif
        xml_path = item['xml'] # Path to .xml
        
        print(f"--> Processing ID: {sample_id}")
        
        # ---------------------------------------------------------
        # INSERT YOUR SEGMENTATION LOGIC HERE
        # You now have the correct paths for the images and metadata
        # ---------------------------------------------------------
        # Example:
        # meta_data = parse_xml(xml_path) 
        # process_image(bf_path, meta_data)
        # ---------------------------------------------------------

    print("Pipeline completed.")

if __name__ == "__main__":
    # This allows you to run it exactly as you did before
    target = input("Enter the name of the folder to process (10 to 19): ")
    
    # Clean input if user types "source/12" instead of just "12"
    if "source/" in target or "source\\" in target:
        target = os.path.basename(target)
        
    run_pipeline(target)