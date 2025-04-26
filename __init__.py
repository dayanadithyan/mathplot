# __init__.py - Main Add-on Registration

bl_info = {
    "name": "Math Playground",
    "author": "Mathematical Explorer",
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Math Playground",
    "description": "Explore mathematical concepts in Blender",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import bpy
import importlib
import sys
import os

# Add the directory to sys.path to enable relative imports
# This is necessary for the add-on to work when installed
__file_path__ = os.path.dirname(__file__)
if __file_path__ not in sys.path:
    sys.path.append(__file_path__)

# Import submodules
from . import properties
from . import operators
from . import ui
from . import utils
from . import algorithms

# Force reload in case of development
if "bpy" in locals():
    importlib.reload(properties)
    importlib.reload(operators)
    importlib.reload(ui)
    importlib.reload(utils)
    importlib.reload(algorithms)

def register():
    """Register the add-on and all its components"""
    # Register properties
    properties.register()
    
    # Register operators
    operators.register()
    
    # Register UI
    ui.register()
    
    # Register utils (if needed)
    utils.register()
    
    # Register algorithms
    algorithms.register()
    
    print("Math Playground: All components registered successfully!")

def unregister():
    """Unregister the add-on and all its components"""
    # Unregister in reverse order
    algorithms.unregister()
    utils.unregister()
    ui.unregister()
    operators.unregister()
    properties.unregister()
    
    print("Math Playground: All components unregistered successfully!")

if __name__ == "__main__":
    register()