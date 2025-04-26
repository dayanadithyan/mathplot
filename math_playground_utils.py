# utils/__init__.py - Utilities for Math Playground

import bpy
import importlib

# Import submodules
from . import materials
from . import collections
from . import progress
from . import math_utils
from . import instancing
from . import exporters

# Force reload in case of development
if "bpy" in locals():
    importlib.reload(materials)
    importlib.reload(collections)
    importlib.reload(progress)
    importlib.reload(math_utils)
    importlib.reload(instancing)
    importlib.reload(exporters)

# Function to get add-on preferences
def get_preferences():
    """Get the add-on preferences"""
    import bpy
    addon_name = __package__.split('.')[0]
    return bpy.context.preferences.addons[addon_name].preferences

def register():
    """Register utility modules"""
    materials.register()
    collections.register()
    progress.register()
    math_utils.register()
    instancing.register()
    exporters.register()

def unregister():
    """Unregister utility modules"""
    exporters.unregister()
    instancing.unregister()
    math_utils.unregister()
    progress.unregister()
    collections.unregister()
    materials.unregister()
