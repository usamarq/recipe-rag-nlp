#!/usr/bin/env python3

"""
Data Downloader Script

This script checks for a target directory (./data) and a specific file
(hummus_recipes.csv). If the directory doesn't exist, it's created.
If the file doesn't exist, it's downloaded from a Google Drive URL
using the 'gdown' library.
"""

import os
import gdown
import sys

def main():
    """
    Main function to check for and download the dataset.
    """
    
    # --- Configuration ---
    # The Google Drive link to your file
    url = 'https://drive.google.com/uc?id=1vBE48dpWDXARqMRxtzdqwJw1wVkSa8yz'
    
    # Define the folder and filename for the output
    output_folder = 'data'
    file_name = 'hummus_recipes.csv'
    destination_path = os.path.join(output_folder, file_name)

    print("--- Starting data check & download process ---")

    # 1. Check if the output folder exists, create it if it doesn't
    try:
        if not os.path.exists(output_folder):
            print(f"Directory '{output_folder}' not found. Creating it.")
            os.makedirs(output_folder)
        else:
            print(f"Directory '{output_folder}' already exists.")
    except OSError as e:
        print(f"❌ Error creating directory '{output_folder}': {e}")
        sys.exit(1) # Exit if we can't create the directory

    # 2. Check if the file already exists, download it if it doesn't
    if not os.path.exists(destination_path):
        print(f"File '{file_name}' not found. Starting download...")
        try:
            # Use gdown to download the file from the URL to the destination path
            gdown.download(url, destination_path, quiet=False)
            print(f"✅ Download complete! File saved to '{destination_path}'")
        except Exception as e:
            print(f"❌ Error during download: {e}")
            # Clean up partial file if download failed
            if os.path.exists(destination_path):
                os.remove(destination_path)
            sys.exit(1) # Exit if download fails
    else:
        print(f"File '{file_name}' already exists in '{output_folder}'. Skipping download.")

    print("--- Process finished successfully ---")

if __name__ == "__main__":
    # This block ensures that main() is called only when
    # the script is executed directly (not when imported)
    main()