# ui/panels.py - UI panel definitions

import bpy
from bpy.types import Panel

# Extract and adapt the Panel classes from math_playground_fixed.py
# MATH_PT_Main, MATH_PT_LinearAlgebra, MATH_PT_NumberTheory, MATH_PT_Analysis, MATH_PT_GraphTheory

class MATH_PT_Main(Panel):
    """Main panel for Math Playground"""
    bl_label = "Math Playground"
    bl_idname = "MATH_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground
        
        layout.label(text="Choose a module:")
        
        # Create buttons for different modules
        box = layout.box()
        col = box.column(align=True)
        
        col.prop(props, "active_module", text="")
        
        # Add a general cleanup button
        layout.separator()
        layout.operator("math.clear_all", text="Clear All Math Objects")

class MATH_PT_LinearAlgebra(Panel):
    """Panel for Linear Algebra tools"""
    bl_label = "Linear Algebra"
    bl_idname = "MATH_PT_linear_algebra"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    bl_parent_id = "MATH_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    @classmethod
    def poll(cls, context):
        return context.scene.math_playground.active_module == 'LINEAR_ALGEBRA'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground.linear_algebra
        
        # Vector properties
        box = layout.box()
        box.label(text="Add Vector")
        col = box.column(align=True)
        col.prop(props, "vector_color", text="Vector Color")
        op = col.operator("math.add_vector", text="Add Vector")
        
        # Matrix properties
        box = layout.box()
        box.label(text="Apply Matrix")
        col = box.column(align=True)
        col.prop(props, "matrix_input", text="Matrix")
        op = col.operator("math.apply_matrix", text="Apply Matrix")
        
        # Clear button
        layout.operator("math.clear_vectors", text="Clear Vectors")

# Add similar Panel classes for NumberTheory, Analysis, GraphTheory, etc.

# Registration functions
classes = [
    MATH_PT_Main,
    MATH_PT_LinearAlgebra,
    # Add other panel classes
]

def register():
    """Register UI panels"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister UI panels"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)