# operators/linear_algebra.py - Linear Algebra operators

import bpy
import math
import numpy as np
from bpy.types import Operator
from mathutils import Vector, Matrix
from ..utils import materials, collections, progress, math_utils

class MATH_OT_AddVector(Operator):
    """Add a vector to the scene"""
    bl_idname = "math.add_vector"
    bl_label = "Add Vector"
    bl_options = {'REGISTER', 'UNDO'}
    
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
        # Create vector collection if it doesn't exist
        collection = collections.get_collection("Math_LinearAlgebra/Math_Vectors")
        
        # Create vector material
        material = materials.create_material(f"Vector_{self.name}_Material", self.color)
        
        # Create the vector object
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=1.0,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            scale=(1, 1, 1)
        )
        vector_obj = bpy.context.active_object
        
        # Set vector length and orientation
        vec = Vector(self.vector)
        length = vec.length
        
        if length > 0:
            # Scale the cylinder to the vector length
            vector_obj.scale.z = length
            
            # Align the cylinder to the vector direction
            if vec.x != 0 or vec.y != 0 or vec.z != 0:
                # Create a rotation from the Z axis to the vector direction
                z_axis = Vector((0, 0, 1))
                vec.normalize()
                rotation_axis = z_axis.cross(vec)
                
                if rotation_axis.length > 0:
                    rotation_angle = math.acos(z_axis.dot(vec))
                    rotation_axis.normalize()
                    vector_obj.rotation_euler = rotation_axis.to_track_quat('Z', 'Y').to_euler()
        
        # Move the base of the cylinder to the origin
        vector_obj.location = Vector(self.vector) / 2
        
        # Add arrow head (cone)
        bpy.ops.mesh.primitive_cone_add(
            radius1=0.05,
            radius2=0,
            depth=0.1,
            enter_editmode=False,
            align='WORLD',
            location=self.vector,
            scale=(1, 1, 1)
        )
        arrowhead = bpy.context.active_object
        
        # Align cone to vector direction
        if length > 0:
            if vec.x != 0 or vec.y != 0 or vec.z != 0:
                arrowhead.rotation_euler = vector_obj.rotation_euler
        
        # Name the objects
        vector_obj.name = f"{self.name}_Shaft"
        arrowhead.name = f"{self.name}_Head"
        
        # Apply material
        materials.apply_material(vector_obj, material)
        materials.apply_material(arrowhead, material)
        
        # Add to collection
        if vector_obj.users_collection:
            vector_obj.users_collection[0].objects.unlink(vector_obj)
        if arrowhead.users_collection:
            arrowhead.users_collection[0].objects.unlink(arrowhead)
        
        collection.objects.link(vector_obj)
        collection.objects.link(arrowhead)
        
        # Store vector data as custom properties
        vector_obj["vector"] = self.vector
        vector_obj["vector_name"] = self.name
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        # Initialize with current settings
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
        try:
            # Parse matrix
            rows = self.matrix_rows.split(';')
            if len(rows) != 3:
                self.report({'ERROR'}, "Matrix must have 3 rows")
                return {'CANCELLED'}
                
            matrix_data = []
            for row in rows:
                values = row.split(',')
                if len(values) != 3:
                    self.report({'ERROR'}, "Each row must have 3 values")
                    return {'CANCELLED'}
                matrix_data.append([float(v) for v in values])
            
            matrix = Matrix(matrix_data)
        except ValueError as e:
            self.report({'ERROR'}, f"Invalid matrix value: {e}")
            return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Error parsing matrix: {e}")
            return {'CANCELLED'}
        
        # Get vector collection
        collection = collections.get_collection("Math_LinearAlgebra/Math_Vectors")
        if not collection:
            self.report({'ERROR'}, "No vectors to transform")
            return {'CANCELLED'}
        
        # Get all vector objects
        vector_objects = []
        for obj in collection.objects:
            if "vector" in obj and obj.name.endswith("_Shaft"):
                vector_objects.append(obj)
        
        if not vector_objects:
            self.report({'ERROR'}, "No vectors to transform")
            return {'CANCELLED'}
        
        progress.report_progress(context, 0.0, "Starting matrix transformation...")
        
        # Process each vector
        for i, obj in enumerate(vector_objects):
            # Report progress
            prog = (i + 1) / len(vector_objects)
            progress.report_progress(context, prog, f"Transforming vector {i+1}/{len(vector_objects)}")
            
            # Extract vector data
            vec = Vector(obj["vector"])
            vec_name = obj["vector_name"]
            
            # Apply transformation
            transformed_vec = matrix @ vec
            
            # Find arrow head
            head_name = f"{vec_name}_Head"
            head = None
            for o in collection.objects:
                if o.name == head_name:
                    head = o
                    break
            
            # Delete existing vector
            if head:
                bpy.data.objects.remove(head, do_unlink=True)
            bpy.data.objects.remove(obj, do_unlink=True)
            
            # Create a new vector with the transformed coordinates
            bpy.ops.math.add_vector(
                vector=transformed_vec,
                color=obj.active_material.node_tree.nodes['Principled BSDF'].inputs[0].default_value,
                name=vec_name
            )
        
        progress.end_progress(context)
        self.report({'INFO'}, f"Applied matrix transformation to {len(vector_objects)} vectors")
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