# __init__.py - Main initialization file for Math Playground

bl_info = {
    "name": "Math Playground",
    "author": "DA SURENDRANATHAN",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Math Playground",
    "description": "Mathematical visualization tools for Blender",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

# Standard modules import
import bpy
import importlib
import sys
import os

# Define version
__version__ = ".".join(map(str, bl_info["version"]))

# Ensure the module path is available
def setup_addon_modules():
    """Set up the addon module path."""
    # Get the addon directory
    dirname = os.path.dirname(__file__)
    
    # Add to path if not already there
    if dirname not in sys.path:
        sys.path.append(dirname)

# Check if we need to do a reload
if "properties" in locals():
    # Module reload
    importlib.reload(properties)
    importlib.reload(operators)
    importlib.reload(ui)
    importlib.reload(algorithms)
    importlib.reload(utils)
else:
    # First import
    from mathplot import properties
    from mathplot import operators
    from mathplot import ui
    from mathplot import algorithms
    from mathplot import utils

    # Import specific utility modules first to ensure proper initialization
    from mathplot.utils import import_utils
    from mathplot.utils import error_utils

# Register function with improved error handling
def register():
    """Register the add-on with comprehensive error handling."""
    try:
        # Set up addon modules path
        setup_addon_modules()
        
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
        
        # Load and register all plugins
        utils.import_utils.plugin_interface.register_plugins()
        
        print(f"Math Playground {__version__} registered successfully")
        return True
        
    except Exception as e:
        # Get detailed error information
        import traceback
        error_msg = traceback.format_exc()
        
        # Print error details
        print(f"Error registering Math Playground: {e}")
        print(error_msg)
        
        # Show error in Blender UI if possible
        if hasattr(bpy.context, 'window_manager'):
            def draw_error(self, context):
                self.layout.label(text=f"Error registering Math Playground: {e}")
            
            bpy.context.window_manager.popup_menu(draw_error, title="Registration Error", icon='ERROR')
        
        return False

def unregister():
    """Unregister the add-on with comprehensive error handling."""
    try:
        # Remove the property group
        if hasattr(bpy.types.Scene, 'math_playground'):
            del bpy.types.Scene.math_playground
        
        # Unregister in reverse order
        ui.unregister()
        operators.unregister()
        algorithms.unregister()
        utils.unregister()
        properties.unregister()
        
        print("Math Playground unregistered")
        return True
        
    except Exception as e:
        # Get detailed error information
        import traceback
        error_msg = traceback.format_exc()
        
        # Print error details
        print(f"Error unregistering Math Playground: {e}")
        print(error_msg)
        
        # Show error in Blender UI if possible
        if hasattr(bpy.context, 'window_manager'):
            def draw_error(self, context):
                self.layout.label(text=f"Error unregistering Math Playground: {e}")
            
            bpy.context.window_manager.popup_menu(draw_error, title="Unregistration Error", icon='ERROR')
        
        return False

if __name__ == "__main__":
    register()