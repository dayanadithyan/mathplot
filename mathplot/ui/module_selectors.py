# ui/module_selectors.py - Module selection operators

import bpy
from bpy.types import Operator

# Extract and adapt the module selector operators from math_playground_fixed.py
# WM_OT_MathLinearAlgebra, WM_OT_MathNumberTheory, etc.

class WM_OT_MathLinearAlgebra(Operator):
    """Switch to Linear Algebra module"""
    bl_idname = "wm.math_linear_algebra"
    bl_label = "Linear Algebra"
    
    def execute(self, context):
        context.scene.math_playground.active_module = 'LINEAR_ALGEBRA'
        return {'FINISHED'}

# Add similar operators for other modules

# Registration functions
classes = [
    WM_OT_MathLinearAlgebra,
    # Add other operator classes
]

def register():
    """Register module selector operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister module selector operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)