# __init__.py - Main initialization file for Math Playground

bl_info = {
    "name": "Math Playground",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Math Playground",
    "description": "Mathematical visualization tools for Blender",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import bpy
from . import properties
from . import operators
from . import ui
from . import algorithms
from . import utils

# Define version
__version__ = ".".join(map(str, bl_info["version"]))

def register():
    """Register the add-on"""
    # Register properties first since they're used by operators and UI
    properties.register()
    
    # Register utility functions
    utils.register()
    
    # Register algorithms
    algorithms.register()
    
    # Register operators
    operators.register()
    
    # Register UI components last
    ui.register()
    
    # Create the main property group for scene
    bpy.types.Scene.math_playground = bpy.props.PointerProperty(type=properties.MathPlaygroundPropertyGroup)
    
    print(f"Math Playground {__version__} registered successfully")

def unregister():
    """Unregister the add-on"""
    # Remove the property group
    del bpy.types.Scene.math_playground
    
    # Unregister in reverse order
    ui.unregister()
    operators.unregister()
    algorithms.unregister()
    utils.unregister()
    properties.unregister()
    
    print("Math Playground unregistered")

if __name__ == "__main__":
    register()