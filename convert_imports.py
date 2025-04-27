#!/usr/bin/env python3

import os
import sys
import ast
import re
import shutil
from datetime import datetime
from pathlib import Path

def create_backup(files):
    """Create backups of all Python files"""
    backup_dir = f"import_conversion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    for file_path in files:
        # Get the file name only, not the full path
        dest_path = Path(backup_dir) / file_path.name
        shutil.copy2(file_path, dest_path)
        print(f"  - Backed up {file_path.name}")
    
    return backup_dir

def get_module_path_from_file(file_path):
    """Determine the module path based on file location"""
    # Get the absolute path and normalize it
    abs_path = file_path.resolve()
    
    # Extract the module path
    # This assumes the file is in a structure like mathplot/submodule/file.py
    parts = abs_path.parts
    
    # Try to find 'mathplot' in the path
    try:
        mathplot_index = parts.index('mathplot')
        module_parts = parts[mathplot_index:-1]  # Exclude the filename
        return '.'.join(module_parts)
    except ValueError:
        # If 'mathplot' not found, assume we're in the root of the project
        return 'mathplot'

def process_file(file_path):
    """Process a single Python file to update imports"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False
    
    # Track changes
    changes_made = False
    original_content = content
    
    # Get the module path for this file
    module_path = get_module_path_from_file(file_path)
    
    # Process the file content using regex for better reliability
    # 1. Handle 'from . import x' or 'from . import *'
    for match in re.finditer(r'from\s+\.\s+import\s+([^#\n]+)', content):
        imported_items = match.group(1).strip()
        old_import = match.group(0)
        
        if imported_items == '*':
            new_import = f"# TODO: Replace wildcard import: from {module_path} import specific_items"
        else:
            new_import = f"from {module_path} import {imported_items}"
        
        content = content.replace(old_import, new_import)
        changes_made = True
        print(f"  - Updated: {old_import} -> {new_import}")
    
    # 2. Handle 'from .submodule import x' or 'from .submodule import *'
    for match in re.finditer(r'from\s+\.([a-zA-Z0-9_]+)\s+import\s+([^#\n]+)', content):
        submodule = match.group(1).strip()
        imported_items = match.group(2).strip()
        old_import = match.group(0)
        
        if imported_items == '*':
            new_import = f"# TODO: Replace wildcard import: from {module_path}.{submodule} import specific_items"
        else:
            new_import = f"from {module_path}.{submodule} import {imported_items}"
        
        content = content.replace(old_import, new_import)
        changes_made = True
        print(f"  - Updated: {old_import} -> {new_import}")
    
    # 3. Handle 'from module import *' (non-relative wildcard imports)
    for match in re.finditer(r'from\s+([a-zA-Z0-9_.]+)\s+import\s+\*', content):
        module_name = match.group(1).strip()
        old_import = match.group(0)
        
        # Skip if this is already a 'mathplot' import
        if not module_name.startswith('mathplot'):
            new_import = f"# TODO: Replace wildcard import: from {module_name} import specific_items"
            content = content.replace(old_import, new_import)
            changes_made = True
            print(f"  - Flagged for review: {old_import}")
    
    # Write the modified content back to the file if changes were made
    if changes_made:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  - Successfully updated {file_path.name}")
        except Exception as e:
            print(f"  - Error writing to {file_path}: {e}")
            # Restore original content if write fails
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                print(f"  - Restored original content for {file_path.name}")
            except:
                print(f"  - WARNING: Could not restore original content for {file_path.name}")
            return False
    
    return changes_made

def main():
    # Find all Python files in the current directory and subdirectories
    python_files = list(Path('.').glob('**/*.py'))
    
    # Filter out the conversion script itself
    script_name = Path(__file__).name
    python_files = [f for f in python_files if f.name != script_name]
    
    if not python_files:
        print("No Python files found in the current directory and subdirectories.")
        return
    
    print(f"Found {len(python_files)} Python files to process.")
    
    # Create backups
    backup_dir = create_backup(python_files)
    print(f"Created backups in {backup_dir}")
    
    # Process each file
    changes_count = 0
    for file_path in python_files:
        print(f"Processing {file_path}")
        if process_file(file_path):
            changes_count += 1
    
    print(f"Conversion complete. Updated {changes_count} files.")
    print("Please manually review TODO comments for wildcard imports.")

if __name__ == "__main__":
    main()