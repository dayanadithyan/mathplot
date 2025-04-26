#!/usr/bin/env python3
"""
Packaging script for Math Playground Blender add-on.
This script organizes all necessary files and creates a zip archive ready for installation.
"""

import os
import sys
import shutil
import zipfile
from datetime import datetime

# Configuration
ADDON_NAME = "math_playground"
VERSION = "2.0.0"
OUTPUT_DIR = "dist"
TEMP_DIR = f"{OUTPUT_DIR}/temp"

# Define the file structure
FILES = [
    "__init__.py",
    "properties.py",
    "LICENSE",
    "README.md",
    "utils/__init__.py",
    "utils/materials.py",
    "utils/collections.py",
    "utils/progress.py",
    "utils/math_utils.py",
    "utils/instancing.py",
    "operators/__init__.py",
    "operators/linear_algebra.py",
    "operators/number_theory.py",
    "operators/analysis.py",
    "operators/graph_theory.py",
    "operators/common.py",
    "ui/__init__.py",
    "ui/panels.py",
    "ui/module_selectors.py",
    "algorithms/__init__.py",
    "algorithms/differential.py",
    "algorithms/fourier.py",
    "algorithms/complex.py",
    "algorithms/graph_algorithms.py",
    "images/math_playground_banner.png"
]

def create_directory_structure():
    """Create the temporary directory structure for packaging."""
    print("Creating directory structure...")
    
    # Create base directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Create subdirectories
    directories = set()
    for file_path in FILES:
        directory = os.path.dirname(file_path)
        if directory and directory not in directories:
            full_path = os.path.join(TEMP_DIR, directory)
            os.makedirs(full_path, exist_ok=True)
            directories.add(directory)
    
    print("Directory structure created.")

def copy_files():
    """Copy all necessary files to the temporary directory."""
    print("Copying files...")
    
    for file_path in FILES:
        source = file_path
        destination = os.path.join(TEMP_DIR, file_path)
        
        # Create directory if it doesn't exist
        dest_dir = os.path.dirname(destination)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        try:
            shutil.copy2(source, destination)
            print(f"Copied: {file_path}")
        except FileNotFoundError:
            print(f"Warning: Could not find {source}")
            
            # Create empty file for missing files to maintain structure
            with open(destination, 'w') as f:
                f.write(f"# {os.path.basename(file_path)} - Placeholder\n")
            print(f"Created placeholder for: {file_path}")
    
    print("Files copied.")

def create_zip_archive():
    """Create a ZIP archive of the add-on."""
    print("Creating ZIP archive...")
    
    # Generate zip filename with version and date
    date_str = datetime.now().strftime("%Y%m%d")
    zip_filename = f"{ADDON_NAME}-{VERSION}-{date_str}.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    
    # Create the zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(TEMP_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, TEMP_DIR)
                zipf.write(file_path, arcname)
    
    print(f"ZIP archive created: {zip_path}")
    return zip_path

def cleanup():
    """Clean up temporary directory."""
    print("Cleaning up...")
    shutil.rmtree(TEMP_DIR)
    print("Cleanup completed.")

def main():
    """Main execution function."""
    print(f"Packaging Math Playground v{VERSION}...")
    
    try:
        create_directory_structure()
        copy_files()
        zip_path = create_zip_archive()
        cleanup()
        
        print("\nPackaging completed successfully!")
        print(f"Add-on package: {zip_path}")
        print("To install in Blender: Edit → Preferences → Add-ons → Install...")
        
        return 0
    except Exception as e:
        print(f"Error during packaging: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())