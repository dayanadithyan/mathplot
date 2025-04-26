# operators/linear_algebra.py - Linear Algebra operators

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Vector, Matrix
from ..utils import materials, collections, progress, math_utils

# Extract and adapt the Linear Algebra operator classes from math_playground_fixed.py
# MATH_OT_AddVector, MATH_OT_ApplyMatrix, MATH_OT_ClearVectors

class MATH_OT_AddVector(Operator):
    """Add a vector to the scene"""
    bl_idname = "math.add_vector"
    bl_label = "Add Vector"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Vector properties
    vector: bpy.props.FloatVectorProperty(
        name="Vector",
        description="Vector coordinates (x, y, z)",
        default=(1.0, 1.0, 1.0),
        subtype='XYZ',
    )
    
    color: bpy.props.FloatVectorProperty(
        name="Color",
        description="Vector color",
        default=(1.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    name: bpy.props.StringProperty(
        name="Name",
        description="Vector name",
        default="Vector",
    )
    
    def execute(self, context):
        # Implementation based on MATH_OT_AddVector from math_playground_fixed.py
        # ...
        return {'FINISHED'}
    
    def invoke(self, context, event):
        # Initialize from scene properties
        self.color = context.scene.math_playground.linear_algebra.vector_color
        return self.execute(context)

class MATH_OT_ApplyMatrix(Operator):
    """Apply a matrix transformation to all vectors"""
    bl_idname = "math.apply_matrix"
    bl_label = "Apply Matrix"
    bl_options = {'REGISTER', 'UNDO'}
    
    matrix_rows: bpy.props.StringProperty(
        name="Matrix",
        description="3x3 matrix in format 'a,b,c;d,e,f;g,h,i'",
        default="1,0,0;0,1,0;0,0,1",
    )
    
    def execute(self, context):
        # Implementation based on MATH_OT_ApplyMatrix from math_playground_fixed.py
        # ...
        return {'FINISHED'}
    
    def invoke(self, context, event):
        self.matrix_rows = context.scene.math_playground.linear_algebra.matrix_input
        return self.execute(context)

class MATH_OT_ClearVectors(Operator):
    """Clear all vectors from the scene"""
    bl_idname = "math.clear_vectors"
    bl_label = "Clear Vectors"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        collections.clear_collection("Math_LinearAlgebra/Math_Vectors")
        self.report({'INFO'}, "All vectors cleared")
        return {'FINISHED'}

# Registration functions
classes = [
    MATH_OT_AddVector,
    MATH_OT_ApplyMatrix,
    MATH_OT_ClearVectors,
]

def register():
    """Register Linear Algebra operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister Linear Algebra operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)