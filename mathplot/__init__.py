# __init__.py - Main initialization file for Math Playground

import os
import sys
import importlib
import bpy

bl_info = {
    "name": "mathplot",
    "author": "DA SURENDRANATHAN",
    "version": (1, 1, 0),
    "blender": (4, 4, 0),
    "location": "View3D > Sidebar > mathplot",
    "description": "Mathematical visualization tools for Blender",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}


# Define version
__version__ = ".".join(map(str, bl_info["version"]))

# Ensure the module path is available
def setup_addon_modules():
    """Set up the addon module path."""
    dirname = os.path.dirname(__file__)
    if dirname not in sys.path:
        sys.path.append(dirname)

# Check if we need to do a reload
if "properties" in locals():
    importlib.reload(properties)
    importlib.reload(operators)
    importlib.reload(ui)
    importlib.reload(algorithms)
    importlib.reload(utils)
else:
    from mathplot import properties
    from mathplot import operators
    from mathplot import ui
    from mathplot import algorithms
    from mathplot import utils
    from mathplot.utils import import_utils
    from mathplot.utils import error_utils

def register():
    """Register the add-on with comprehensive error handling."""
    try:
        setup_addon_modules()

        properties.register()
        utils.register()
        algorithms.register()
        operators.register()
        ui.register()

        bpy.types.Scene.math_playground = bpy.props.PointerProperty(
            type=properties.MathPlaygroundPropertyGroup
        )

        utils.import_utils.plugin_interface.register_plugins()

        print(f"Math Playground {__version__} registered successfully")
        return True

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()

        print(f"Error registering Math Playground: {e}")
        print(error_msg)

        if hasattr(bpy.context, 'window_manager'):
            def draw_error(self, context):
                """Draw error popup during registration failure."""
                self.layout.label(text=f"Error registering Math Playground: {e}")

            bpy.context.window_manager.popup_menu(
                draw_error, title="Registration Error", icon='ERROR'
            )

        return False

def unregister():
    """Unregister the add-on with comprehensive error handling."""
    try:
        if hasattr(bpy.types.Scene, 'math_playground'):
            del bpy.types.Scene.math_playground

        ui.unregister()
        operators.unregister()
        algorithms.unregister()
        utils.unregister()
        properties.unregister()

        print("Math Playground unregistered")
        return True

    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()

        print(f"Error unregistering Math Playground: {e}")
        print(error_msg)

        if hasattr(bpy.context, 'window_manager'):
            def draw_error(self, context):
                """Draw error popup during unregistration failure."""
                self.layout.label(text=f"Error unregistering Math Playground: {e}")

            bpy.context.window_manager.popup_menu(
                draw_error, title="Unregistration Error", icon='ERROR'
            )

        return False

if __name__ == "__main__":
    register()
