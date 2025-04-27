# ui/module_selectors.py - Module selection operators

import bpy
from bpy.types import Operator

class WM_OT_MathLinearAlgebra(Operator):
    """Switch to Linear Algebra module"""
    bl_idname = "wm.math_linear_algebra"
    bl_label = "Linear Algebra"
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    context.scene.math_playground.active_module = 'LINEAR_ALGEBRA'
        return {'FINISHED'}

class WM_OT_MathNumberTheory(Operator):
    """Switch to Number Theory module"""
    bl_idname = "wm.math_number_theory"
    bl_label = "Number Theory"
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    context.scene.math_playground.active_module = 'NUMBER_THEORY'
        return {'FINISHED'}

class WM_OT_MathAnalysis(Operator):
    """Switch to Analysis module"""
    bl_idname = "wm.math_analysis"
    bl_label = "Analysis"
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    context.scene.math_playground.active_module = 'ANALYSIS'
        return {'FINISHED'}

class WM_OT_MathGraphTheory(Operator):
    """Switch to Graph Theory module"""
    bl_idname = "wm.math_graph_theory"
    bl_label = "Graph Theory"
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    context.scene.math_playground.active_module = 'GRAPH_THEORY'
        return {'FINISHED'}

# Registration functions
classes = [
    WM_OT_MathLinearAlgebra,
    WM_OT_MathNumberTheory,
    WM_OT_MathAnalysis,
    WM_OT_MathGraphTheory,
]

def register():
    """Register module selector operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister module selector operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)