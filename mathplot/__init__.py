bl_info = {
    "name": "mathplot",
    "author": "your-name",
    "version": (2, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Tool Shelf",
    "description": "a collection of math visualization tools",
    "warning": "",
    "wiki_url": "",
    "category": "Development",
}

# import submodules from the local package
from .utils.collections import CollectionHelpers
from .math_utils import MathUtils
from .progress import ProgressTracker
from .properties import MathPlotProperties

# import blender operators, ui, algorithms, utils
from .operators import register as register_operators, unregister as unregister_operators
from .ui import register as register_ui, unregister as unregister_ui
from .algorithms import register as register_algorithms, unregister as unregister_algorithms
from .utils import register as register_utils, unregister as unregister_utils

def register():
    # register core modules if they need Blender registration hooks
    MathPlotProperties.register()
    ProgressTracker.register()
    CollectionHelpers.register()
    MathUtils.register()
    # register subpackages
    register_utils()
    register_operators()
    register_ui()
    register_algorithms()

def unregister():
    # unregister in reverse order
    unregister_algorithms()
    unregister_ui()
    unregister_operators()
    unregister_utils()
    MathUtils.unregister()
    CollectionHelpers.unregister()
    ProgressTracker.unregister()
    MathPlotProperties.unregister()
