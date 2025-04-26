class MATH_OT_ClearGraphs(Operator):
    """Clear all graph objects from the scene"""
    bl_idname = "math.clear_graphs"
    bl_label = "Clear Graphs"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Graphs")
        self.report({'INFO'}, "All graph objects cleared")
        return {'FINISHED'}

class MATH_OT_ClearAll(Operator):
    """Clear all math objects from the scene"""
    bl_idname = "math.clear_all"
    bl_label = "Clear All Math Objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Clear all math collections
        clear_collection("Math_Vectors")
        clear_collection("Math_Primes")
        clear_collection("Math_Sequences")
        clear_collection("Math_Functions")
        clear_collection("Math_VectorField")
        clear_collection("Math_Graphs")
        
        self.report({'INFO'}, "All math objects cleared")
        return {'FINISHED'}

# ------------------------------------------------
# UI Panels
# ------------------------------------------------

class MATH_PT_Main(Panel):
    """Main panel for Math Playground"""
    bl_label = "Math Playground"
    bl_idname = "MATH_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    
    def draw(self, context):
        layout = self.layout
        
        layout.label(text="Choose a module:")
        
        # Create buttons for different modules
        box = layout.box()
        col = box.column(align=True)
        
        col.operator("wm.math_linear_algebra", text="Linear Algebra")
        col.operator("wm.math_number_theory", text="Number Theory")
        col.operator("wm.math_analysis", text="Analysis")
        col.operator("wm.math_graph_theory", text="Graph Theory")
        
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
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground
        
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

class MATH_PT_NumberTheory(Panel):
    """Panel for Number Theory tools"""
    bl_label = "Number Theory"
    bl_idname = "MATH_PT_number_theory"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    bl_parent_id = "MATH_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground
        
        # Prime generator
        box = layout.box()
        box.label(text="Generate Primes")
        col = box.column(align=True)
        col.prop(props, "prime_limit", text="Up to")
        op = col.operator("math.generate_primes", text="Generate Primes")
        
        # Sequence generator
        box = layout.box()
        box.label(text="Generate Sequence")
        col = box.column(align=True)
        col.prop(props, "sequence_length", text="Length")
        op = col.operator("math.generate_sequence", text="Generate Sequence")
        
        # Clear button
        layout.operator("math.clear_number_theory", text="Clear Number Theory")

class MATH_PT_Analysis(Panel):
    """Panel for Analysis tools"""
    bl_label = "Analysis"
    bl_idname = "MATH_PT_analysis"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    bl_parent_id = "MATH_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground
        
        # Function plotter
        box = layout.box()
        box.label(text="Plot Function")
        col = box.column(align=True)
        col.prop(props, "function_expression", text="f(x)")
        col.prop(props, "x_min", text="X Min")
        col.prop(props, "x_max", text="X Max")
        op = col.operator("math.plot_function", text="Plot Function")
        
        # Parametric curve plotter
        box = layout.box()
        box.label(text="Plot Parametric Curve")
        col = box.column(align=True)
        op = col.operator("math.plot_parametric", text="Plot Parametric Curve")
        
        # Vector field plotter
        box = layout.box()
        box.label(text="Plot Vector Field")
        col = box.column(align=True)
        op = col.operator("math.plot_vector_field", text="Plot Vector Field")
        
        # Clear button
        layout.operator("math.clear_analysis", text="Clear Analysis")

class MATH_PT_GraphTheory(Panel):
    """Panel for Graph Theory tools"""
    bl_label = "Graph Theory"
    bl_idname = "MATH_PT_graph_theory"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    bl_parent_id = "MATH_PT_main"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.math_playground
        
        # Graph creator
        box = layout.box()
        box.label(text="Create Graph")
        col = box.column(align=True)
        col.prop(props, "num_nodes", text="Nodes")
        col.prop(props, "edge_probability", text="Edge Probability")
        op = col.operator("math.create_graph", text="Create Graph")
        
        # Graph algorithms
        box = layout.box()
        box.label(text="Run Algorithm")
        col = box.column(align=True)
        op = col.operator("math.run_graph_algorithm", text="Run Algorithm")
        
        # Clear button
        layout.operator("math.clear_graphs", text="Clear Graphs")

# ------------------------------------------------
# Module Selector Operators
# ------------------------------------------------

class WM_OT_MathLinearAlgebra(Operator):
    """Switch to Linear Algebra module"""
    bl_idname = "wm.math_linear_algebra"
    bl_label = "Linear Algebra"
    
    def execute(self, context):
        bpy.ops.wm.context_set_string(data_path="space_data.context", value="MATH_PT_linear_algebra")
        return {'FINISHED'}

class WM_OT_MathNumberTheory(Operator):
    """Switch to Number Theory module"""
    bl_idname = "wm.math_number_theory"
    bl_label = "Number Theory"
    
    def execute(self, context):
        bpy.ops.wm.context_set_string(data_path="space_data.context", value="MATH_PT_number_theory")
        return {'FINISHED'}

class WM_OT_MathAnalysis(Operator):
    """Switch to Analysis module"""
    bl_idname = "wm.math_analysis"
    bl_label = "Analysis"
    
    def execute(self, context):
        bpy.ops.wm.context_set_string(data_path="space_data.context", value="MATH_PT_analysis")
        return {'FINISHED'}

class WM_OT_MathGraphTheory(Operator):
    """Switch to Graph Theory module"""
    bl_idname = "wm.math_graph_theory"
    bl_label = "Graph Theory"
    
    def execute(self, context):
        bpy.ops.wm.context_set_string(data_path="space_data.context", value="MATH_PT_graph_theory")
        return {'FINISHED'}

# ------------------------------------------------
# Add-on Preferences
# ------------------------------------------------

class MathPlaygroundPreferences(AddonPreferences):
    """Preferences for Math Playground add-on"""
    bl_idname = __name__
    
    show_debug_info: BoolProperty(
        name="Show Debug Info",
        description="Show additional debug information in the console",
        default=False
    )
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "show_debug_info")
        layout.label(text="Math Playground Add-on v1.0.0")
        layout.label(text="Explore mathematical concepts in Blender's 3D environment")
        layout.separator()
        layout.label(text="Documentation:")
        box = layout.box()
        box.label(text="Linear Algebra: Create vectors and apply matrix transformations")
        box.label(text="Number Theory: Visualize prime numbers and sequences")
        box.label(text="Analysis: Plot functions and vector fields")
        box.label(text="Graph Theory: Create graphs and run algorithms on them")

# ------------------------------------------------
# Registration
# ------------------------------------------------

classes = (
    # Property groups
    MathPlaygroundProperties,
    
    # Linear Algebra operators
    MATH_OT_AddVector,
    MATH_OT_ApplyMatrix,
    MATH_OT_ClearVectors,
    
    # Number Theory operators
    MATH_OT_GeneratePrimes,
    MATH_OT_GenerateSequence,
    MATH_OT_ClearNumberTheory,
    
    # Analysis operators
    MATH_OT_PlotFunction,
    MATH_OT_PlotParametric,
    MATH_OT_PlotVectorField,
    MATH_OT_ClearAnalysis,
    
    # Graph Theory operators
    MATH_OT_CreateGraph,
    MATH_OT_RunGraphAlgorithm,
    MATH_OT_ClearGraphs,
    
    # General operators
    MATH_OT_ClearAll,
    
    # Module selector operators
    WM_OT_MathLinearAlgebra,
    WM_OT_MathNumberTheory,
    WM_OT_MathAnalysis,
    WM_OT_MathGraphTheory,
    
    # Panels
    MATH_PT_Main,
    MATH_PT_LinearAlgebra,
    MATH_PT_NumberTheory,
    MATH_PT_Analysis,
    MATH_PT_GraphTheory,
    
    # Preferences
    MathPlaygroundPreferences,
)

def register():
    """Register all classes and properties"""
    # Register classes
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register properties
    bpy.types.Scene.math_playground = bpy.props.PointerProperty(type=MathPlaygroundProperties)

def unregister():
    """Unregister all classes and properties"""
    # Unregister properties
    del bpy.types.Scene.math_playground
    
    # Unregister classes (in reverse order)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    # Clean up collections
    try:
        clear_collection("Math_Vectors")
        clear_collection("Math_Primes")
        clear_collection("Math_Sequences")
        clear_collection("Math_Functions")
        clear_collection("Math_VectorField")
        clear_collection("Math_Graphs")
    except:
        pass

if __name__ == "__main__":
    register()"""
Math Playground Add-on for Blender
===================================

A comprehensive mathematical playground for exploring concepts in Blender's 3D environment.

Features:
- Linear Algebra: vectors, matrices, transformations
- Number Theory: prime numbers, integer sequences
- Analysis: function plotting, vector fields
- Graph Theory: graph visualization, algorithms

Author: Mathematical Explorer
Version: 1.0.0
Blender: 3.0.0
"""

bl_info = {
    "name": "Math Playground",
    "author": "Mathematical Explorer",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Math Playground",
    "description": "Explore mathematical concepts in Blender",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}

import bpy
import numpy as np
import math
import random
from mathutils import Vector, Matrix
from bpy.props import (
    StringProperty, BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, EnumProperty, PointerProperty
)
from bpy.types import (
    Panel, Operator, PropertyGroup, Scene, AddonPreferences
)

# ------------------------------------------------
# Utility Functions
# ------------------------------------------------

def create_material(name, color):
    """Create a new material with the given name and color.
    
    Args:
        name (str): The name for the material
        color (tuple): RGBA color values as a 4-tuple (r, g, b, a)
        
    Returns:
        bpy.types.Material: The created or retrieved material
    """
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    
    material.use_nodes = True
    principled_bsdf = material.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        # Ensure color has 4 components (RGBA)
        if len(color) == 3:
            color = (*color, 1.0)  # Add alpha=1.0 if missing
        principled_bsdf.inputs[0].default_value = color
    
    return material

def apply_material(obj, material):
    """Apply material to an object.
    
    Args:
        obj (bpy.types.Object): The object to apply material to
        material (bpy.types.Material): The material to apply
    """
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)

def generate_random_color():
    """Generate a random RGBA color.
    
    Returns:
        tuple: Random RGBA color as (r, g, b, a)
    """
    return (random.random(), random.random(), random.random(), 1.0)

def get_collection(name):
    """Get or create a collection with the given name.
    
    Args:
        name (str): Name of the collection to get or create
        
    Returns:
        bpy.types.Collection: The retrieved or created collection
    """
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection

def clear_collection(collection_name):
    """Remove all objects from a collection.
    
    Args:
        collection_name (str): Name of the collection to clear
    """
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Use a list to avoid modifying the collection during iteration
        objects_to_remove = [obj for obj in collection.objects]
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)

def report_progress(context, progress, message):
    """Report progress to the user during long operations.
    
    Args:
        context (bpy.types.Context): Current context
        progress (float): Progress value between 0.0 and 1.0
        message (str): Progress message to display
    """
    context.window_manager.progress_update(progress)
    context.window_manager.progress_begin(0, 100)
    context.window_manager.progress_update(int(progress * 100))
    if hasattr(context.area, "header_text_set"):
        context.area.header_text_set(message)
    return

def end_progress(context):
    """End progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
    """
    context.window_manager.progress_end()
    if hasattr(context.area, "header_text_set"):
        context.area.header_text_set(None)
    return

# ------------------------------------------------
# Mathematical Helper Functions
# ------------------------------------------------

def is_prime(n):
    """Determine if a number is prime.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def generate_primes(limit):
    """Generate a list of prime numbers up to a limit.
    
    Args:
        limit (int): Upper limit for prime generation
        
    Returns:
        list: List of prime numbers
    """
    primes = []
    for i in range(2, limit + 1):
        if is_prime(i):
            primes.append(i)
    return primes

def evaluate_expression(expression, variables):
    """Safely evaluate a mathematical expression.
    
    Args:
        expression (str): Mathematical expression as string
        variables (dict): Dictionary of variables to use in evaluation
        
    Returns:
        float: Result of expression evaluation
        
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    # Provide a safe namespace for evaluation
    safe_namespace = {
        "math": math,
        "np": np,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
        **variables
    }
    
    # Check for potentially unsafe operations
    unsafe_terms = ["__", "import", "eval", "exec", "compile", "globals", "locals"]
    for term in unsafe_terms:
        if term in expression:
            raise ValueError(f"Unsafe term detected in expression: {term}")
    
    try:
        return eval(expression, {"__builtins__": {}}, safe_namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

# ------------------------------------------------
# Add-on Properties
# ------------------------------------------------

class MathPlaygroundProperties(PropertyGroup):
    """Property group for persistent settings."""
    
    # Linear Algebra properties
    vector_color: FloatVectorProperty(
        name="Vector Color",
        description="Default color for vectors",
        default=(1.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    matrix_input: StringProperty(
        name="Matrix",
        description="3x3 matrix in format 'a,b,c;d,e,f;g,h,i'",
        default="1,0,0;0,1,0;0,0,1",
    )
    
    # Number Theory properties
    prime_limit: IntProperty(
        name="Prime Limit",
        description="Generate primes up to this number",
        default=100,
        min=2,
        max=10000,
    )
    
    sequence_length: IntProperty(
        name="Sequence Length",
        description="Number of terms to generate in sequences",
        default=10,
        min=1,
        max=100,
    )
    
    # Analysis properties
    function_expression: StringProperty(
        name="Function",
        description="Python expression for y=f(x)",
        default="math.sin(x)",
    )
    
    x_min: FloatProperty(
        name="X Min",
        description="Minimum x value for plotting",
        default=-10.0,
    )
    
    x_max: FloatProperty(
        name="X Max",
        description="Maximum x value for plotting",
        default=10.0,
    )
    
    # Graph Theory properties
    num_nodes: IntProperty(
        name="Number of Nodes",
        description="Number of nodes in the graph",
        default=10,
        min=2,
        max=100,
    )
    
    edge_probability: FloatProperty(
        name="Edge Probability",
        description="Probability of edge connection between nodes",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

# ------------------------------------------------
# Linear Algebra Module
# ------------------------------------------------

class MATH_OT_AddVector(Operator):
    """Add a vector to the scene"""
    bl_idname = "math.add_vector"
    bl_label = "Add Vector"
    bl_options = {'REGISTER', 'UNDO'}
    
    vector: FloatVectorProperty(
        name="Vector",
        description="Vector coordinates (x, y, z)",
        default=(1.0, 1.0, 1.0),
        subtype='XYZ',
    )
    
    color: FloatVectorProperty(
        name="Color",
        description="Vector color",
        default=(1.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    name: StringProperty(
        name="Name",
        description="Vector name",
        default="Vector",
    )
    
    def execute(self, context):
        # Create vector collection if it doesn't exist
        collection = get_collection("Math_Vectors")
        
        # Create vector material
        material = create_material(f"Vector_{self.name}_Material", self.color)
        
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
        apply_material(vector_obj, material)
        apply_material(arrowhead, material)
        
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
        self.color = context.scene.math_playground.vector_color
        return self.execute(context)

class MATH_OT_ApplyMatrix(Operator):
    """Apply a matrix transformation to all vectors"""
    bl_idname = "math.apply_matrix"
    bl_label = "Apply Matrix"
    bl_options = {'REGISTER', 'UNDO'}
    
    matrix_rows: StringProperty(
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
        collection = bpy.data.collections.get("Math_Vectors")
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
        
        report_progress(context, 0.0, "Starting matrix transformation...")
        
        # Process each vector
        for i, obj in enumerate(vector_objects):
            # Report progress
            progress = (i + 1) / len(vector_objects)
            report_progress(context, progress, f"Transforming vector {i+1}/{len(vector_objects)}")
            
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
        
        end_progress(context)
        self.report({'INFO'}, f"Applied matrix transformation to {len(vector_objects)} vectors")
        return {'FINISHED'}
    
    def invoke(self, context, event):
        # Initialize with current settings
        self.matrix_rows = context.scene.math_playground.matrix_input
        return self.execute(context)

class MATH_OT_ClearVectors(Operator):
    """Clear all vectors from the scene"""
    bl_idname = "math.clear_vectors"
    bl_label = "Clear Vectors"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Vectors")
        self.report({'INFO'}, "All vectors cleared")
        return {'FINISHED'}

# ------------------------------------------------
# Number Theory Module
# ------------------------------------------------

class MATH_OT_GeneratePrimes(Operator):
    """Generate prime numbers up to a limit"""
    bl_idname = "math.generate_primes"
    bl_label = "Generate Primes"
    bl_options = {'REGISTER', 'UNDO'}
    
    limit: IntProperty(
        name="Limit",
        description="Generate primes up to this number",
        default=100,
        min=2,
        max=10000,
    )
    
    arrangement: EnumProperty(
        name="Arrangement",
        description="How to arrange the prime numbers",
        items=[
            ('LINE', "Line", "Arrange primes in a line"),
            ('SPIRAL', "Ulam Spiral", "Arrange primes in an Ulam spiral"),
            ('GRID', "Grid", "Arrange primes in a grid")
        ],
        default='LINE',
    )
    
    radius: FloatProperty(
        name="Sphere Radius",
        description="Radius of the spheres representing primes",
        default=0.1,
        min=0.01,
        max=1.0,
    )
    
    spacing: FloatProperty(
        name="Spacing",
        description="Spacing between spheres",
        default=0.5,
        min=0.1,
        max=5.0,
    )
    
    def execute(self, context):
        # Create primes collection if it doesn't exist
        collection = get_collection("Math_Primes")
        
        # Clear existing primes
        clear_collection("Math_Primes")
        
        # Generate primes
        try:
            report_progress(context, 0.0, f"Generating primes up to {self.limit}...")
            primes = generate_primes(self.limit)
            
            if not primes:
                self.report({'WARNING'}, f"No primes found up to {self.limit}")
                end_progress(context)
                return {'CANCELLED'}
            
            # Create material for primes
            prime_material = create_material("Prime_Material", (0.0, 0.8, 0.2, 1.0))
            
            # Place primes based on arrangement
            if self.arrangement == 'LINE':
                self.create_line_arrangement(context, primes, prime_material, collection)
            elif self.arrangement == 'SPIRAL':
                self.create_spiral_arrangement(context, primes, prime_material, collection)
            elif self.arrangement == 'GRID':
                self.create_grid_arrangement(context, primes, prime_material, collection)
            
            end_progress(context)
            self.report({'INFO'}, f"Generated {len(primes)} prime numbers")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error generating primes: {e}")
            return {'CANCELLED'}
    
    def create_line_arrangement(self, context, primes, material, collection):
        """Arrange primes in a line"""
        for i, p in enumerate(primes):
            # Report progress
            progress = (i + 1) / len(primes)
            report_progress(context, progress, f"Creating prime {i+1}/{len(primes)}")
            
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=self.radius,
                location=(i * self.spacing, 0, 0),
                segments=16, 
                ring_count=8
            )
            sphere = bpy.context.active_object
            sphere.name = f"Prime_{p}"
            
            # Add text to show the prime number
            bpy.ops.object.text_add(location=(i * self.spacing, 0, self.radius + 0.1))
            text = bpy.context.active_object
            text.data.body = str(p)
            text.data.align_x = 'CENTER'
            text.rotation_euler.x = math.pi / 2
            text.name = f"PrimeText_{p}"
            
            # Apply material
            apply_material(sphere, material)
            
            # Add to collection
            if sphere.users_collection:
                sphere.users_collection[0].objects.unlink(sphere)
            if text.users_collection:
                text.users_collection[0].objects.unlink(text)
            
            collection.objects.link(sphere)
            collection.objects.link(text)
    
    def create_spiral_arrangement(self, context, primes, material, collection):
        """Arrange primes in an Ulam spiral"""
        # Determine size of the spiral grid
        size = math.ceil(math.sqrt(self.limit))
        if size % 2 == 0:
            size += 1  # Ensure odd size for centered spiral
            
        # Create grid of numbers
        grid = np.zeros((size, size), dtype=int)
        center = size // 2
        
        x, y = center, center
        num = 1
        grid[y][x] = num
        
        # Fill the spiral grid with consecutive integers
        for layer in range(1, size // 2 + 1):
            # Move right
            x += 1
            num += 1
            grid[y][x] = num
            
            # Move up
            for _ in range(2 * layer - 1):
                y -= 1
                num += 1
                if num <= self.limit:
                    grid[y][x] = num
            
            # Move left
            for _ in range(2 * layer):
                x -= 1
                num += 1
                if num <= self.limit:
                    grid[y][x] = num
            
            # Move down
            for _ in range(2 * layer):
                y += 1
                num += 1
                if num <= self.limit:
                    grid[y][x] = num
            
            # Move right
            for _ in range(2 * layer):
                x += 1
                num += 1
                if num <= self.limit:
                    grid[y][x] = num
            
            if num >= self.limit:
                break
        
        # Place spheres at prime positions
        prime_count = 0
        total_primes = len(primes)
        
        for y in range(size):
            for x in range(size):
                n = grid[y][x]
                if n > 0 and n <= self.limit and is_prime(n):
                    # Report progress
                    prime_count += 1
                    report_progress(context, prime_count / total_primes, 
                                    f"Creating prime {prime_count}/{total_primes}")
                    
                    # Create sphere
                    bpy.ops.mesh.primitive_uv_sphere_add(
                        radius=self.radius,
                        location=((x - center) * self.spacing, (center - y) * self.spacing, 0),
                        segments=16, 
                        ring_count=8
                    )
                    sphere = bpy.context.active_object
                    sphere.name = f"Prime_{n}"
                    
                    # Add text to show the prime number
                    bpy.ops.object.text_add(
                        location=((x - center) * self.spacing, (center - y) * self.spacing, self.radius + 0.1)
                    )
                    text = bpy.context.active_object
                    text.data.body = str(n)
                    text.data.align_x = 'CENTER'
                    text.rotation_euler.x = math.pi / 2
                    text.name = f"PrimeText_{n}"
                    
                    # Apply material
                    apply_material(sphere, material)
                    
                    # Add to collection
                    if sphere.users_collection:
                        sphere.users_collection[0].objects.unlink(sphere)
                    if text.users_collection:
                        text.users_collection[0].objects.unlink(text)
                    
                    collection.objects.link(sphere)
                    collection.objects.link(text)
    
    def create_grid_arrangement(self, context, primes, material, collection):
        """Arrange primes in a grid"""
        cols = math.ceil(math.sqrt(len(primes)))
        
        for i, p in enumerate(primes):
            # Report progress
            progress = (i + 1) / len(primes)
            report_progress(context, progress, f"Creating prime {i+1}/{len(primes)}")
            
            row = i // cols
            col = i % cols
            
            # Create sphere
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=self.radius,
                location=(col * self.spacing, -row * self.spacing, 0),
                segments=16, 
                ring_count=8
            )
            sphere = bpy.context.active_object
            sphere.name = f"Prime_{p}"
            
            # Add text to show the prime number
            bpy.ops.object.text_add(location=(col * self.spacing, -row * self.spacing, self.radius + 0.1))
            text = bpy.context.active_object
            text.data.body = str(p)
            text.data.align_x = 'CENTER'
            text.rotation_euler.x = math.pi / 2
            text.name = f"PrimeText_{p}"
            
            # Apply material
            apply_material(sphere, material)
            
            # Add to collection
            if sphere.users_collection:
                sphere.users_collection[0].objects.unlink(sphere)
            if text.users_collection:
                text.users_collection[0].objects.unlink(text)
            
            collection.objects.link(sphere)
            collection.objects.link(text)
    
    def invoke(self, context, event):
        # Initialize with current settings
        self.limit = context.scene.math_playground.prime_limit
        return self.execute(context)

class MATH_OT_GenerateSequence(Operator):
    """Generate integer sequences"""
    bl_idname = "math.generate_sequence"
    bl_label = "Generate Sequence"
    bl_options = {'REGISTER', 'UNDO'}
    
    sequence_type: EnumProperty(
        name="Sequence Type",
        description="Type of integer sequence to generate",
        items=[
            ('FIBONACCI', "Fibonacci", "Fibonacci sequence"),
            ('SQUARE', "Square Numbers", "Square numbers"),
            ('TRIANGULAR', "Triangular Numbers", "Triangular numbers"),
            ('PRIME', "Prime Numbers", "Prime numbers"),
            ('CUSTOM', "Custom Formula", "Custom sequence formula")
        ],
        default='FIBONACCI',
    )
    
    length: IntProperty(
        name="Length",
        description="Number of terms to generate",
        default=10,
        min=1,
        max=100,
    )
    
    custom_formula: StringProperty(
        name="Custom Formula",
        description="Python expression for custom sequence (use n for term index)",
        default="n**2 + 1",
    )
    
    cube_size: FloatProperty(
        name="Cube Size",
        description="Size of cubes representing sequence terms",
        default=0.5,
        min=0.1,
        max=5.0,
    )
    
    spacing: FloatProperty(
        name="Spacing",
        description="Spacing between cubes",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    
    height_scaling: FloatProperty(
        name="Height Scaling",
        description="Scale factor for term value to cube height",
        default=0.1,
        min=0.01,
        max=1.0,
    )
    
    def execute(self, context):
        # Create sequence collection if it doesn't exist
        collection = get_collection("Math_Sequences")
        
        # Clear existing sequence
        clear_collection("Math_Sequences")
        
        try:
            # Generate sequence
            report_progress(context, 0.0, f"Generating {self.sequence_type} sequence...")
            sequence = self.generate_sequence()
            
            if not sequence:
                self.report({'WARNING'}, "Failed to generate sequence")
                end_progress(context)
                return {'CANCELLED'}
            
            # Create material for sequence
            sequence_material = create_material(
                f"{self.sequence_type}_Sequence_Material", 
                (0.2, 0.4, 0.8, 1.0)
            )
            
            # Place cubes
            for i, term in enumerate(sequence):
                # Report progress
                progress = (i + 1) / len(sequence)
                report_progress(context, progress, f"Creating term {i+1}/{len(sequence)}")
                
                # Calculate height based on term value
                height = max(0.1, term * self.height_scaling)
                
                # Create cube
                bpy.ops.mesh.primitive_cube_add(
                    size=self.cube_size,
                    enter_editmode=False,
                    align='WORLD',
                    location=(i * self.spacing, 0, height / 2),
                    scale=(1, 1, height / self.cube_size)
                )
                cube = bpy.context.active_object
                cube.name = f"Sequence_{self.sequence_type}_{i+1}_{term}"
                
                # Add text to show the term value
                bpy.ops.object.text_add(location=(i * self.spacing, 0, height + 0.2))
                text = bpy.context.active_object
                text.data.body = str(term)
                text.data.align_x = 'CENTER'
                text.rotation_euler.x = math.pi / 2
                text.name = f"SequenceText_{i+1}_{term}"
                
                # Apply material
                apply_material(cube, sequence_material)
                
                # Add to collection
                if cube.users_collection:
                    cube.users_collection[0].objects.unlink(cube)
                if text.users_collection:
                    text.users_collection[0].objects.unlink(text)
                
                collection.objects.link(cube)
                collection.objects.link(text)
            
            end_progress(context)
            self.report({'INFO'}, f"Generated {len(sequence)} terms of {self.sequence_type} sequence")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error generating sequence: {e}")
            return {'CANCELLED'}
    
    def generate_sequence(self):
        """Generate the requested sequence"""
        sequence = []
        
        if self.sequence_type == 'FIBONACCI':
            # Fibonacci sequence
            a, b = 0, 1
            for _ in range(self.length):
                sequence.append(a)
                a, b = b, a + b
        
        elif self.sequence_type == 'SQUARE':
            # Square numbers
            sequence = [n**2 for n in range(1, self.length + 1)]
        
        elif self.sequence_type == 'TRIANGULAR':
            # Triangular numbers
            sequence = [n * (n + 1) // 2 for n in range(1, self.length + 1)]
        
        elif self.sequence_type == 'PRIME':
            # Prime numbers
            sequence = []
            n = 2
            while len(sequence) < self.length:
                if is_prime(n):
                    sequence.append(n)
                n += 1
        
        elif self.sequence_type == 'CUSTOM':
            # Custom formula
            try:
                sequence = []
                for n in range(1, self.length + 1):
                    value = evaluate_expression(self.custom_formula, {"n": n})
                    sequence.append(value)
            except ValueError as e:
                raise ValueError(f"Error in custom formula: {e}")
        
        return sequence
    
    def invoke(self, context, event):
        # Initialize with current settings
        self.length = context.scene.math_playground.sequence_length
        return self.execute(context)

class MATH_OT_ClearNumberTheory(Operator):
    """Clear all number theory objects from the scene"""
    bl_idname = "math.clear_number_theory"
    bl_label = "Clear Number Theory"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Primes")
        clear_collection("Math_Sequences")
        self.report({'INFO'}, "All number theory objects cleared")
        return {'FINISHED'}

# ------------------------------------------------
# Analysis Module
# ------------------------------------------------

class MATH_OT_PlotFunction(Operator):
    """Plot a function y=f(x)"""
    bl_idname = "math.plot_function"
    bl_label = "Plot Function"
    bl_options = {'REGISTER', 'UNDO'}
    
    function: StringProperty(
        name="Function",
        description="Python expression for y=f(x)",
        default="math.sin(x)",
    )
    
    x_min: FloatProperty(
        name="X Min",
        description="Minimum x value",
        default=-10.0,
    )
    
    x_max: FloatProperty(
        name="X Max",
        description="Maximum x value",
        default=10.0,
    )
    
    samples: IntProperty(
        name="Samples",
        description="Number of sample points",
        default=100,
        min=10,
        max=1000,
    )
    
    thickness: FloatProperty(
        name="Curve Thickness",
        description="Thickness of the curve",
        default=0.05,
        min=0.01,
        max=0.5,
    )
    
    color: FloatVectorProperty(
        name="Color",
        description="Curve color",
        default=(0.0, 0.6, 0.8, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    plot_type: EnumProperty(
        name="Plot Type",
        description="Type of plot to create",
        items=[
            ('CURVE', "Curve", "Plot as a curve"),
            ('SURFACE', "Surface", "Plot as a 3D surface"),
        ],
        default='CURVE',
    )
    
    z_function: StringProperty(
        name="Z Function",
        description="Python expression for z=f(x,y) (for surface plots)",
        default="math.sin(math.sqrt(x**2 + y**2))",
    )
    
    y_min: FloatProperty(
        name="Y Min",
        description="Minimum y value (for surface plots)",
        default=-10.0,
    )
    
    y_max: FloatProperty(
        name="Y Max",
        description="Maximum y value (for surface plots)",
        default=10.0,
    )
    
    def execute(self, context):
        # Create function collection if it doesn't exist
        collection = get_collection("Math_Functions")
        
        # Create material for function
        func_material = create_material(f"Function_Material", self.color)
        
        try:
            report_progress(context, 0.0, "Starting function plot...")
            
            if self.plot_type == 'CURVE':
                self.plot_curve(context, func_material, collection)
            elif self.plot_type == 'SURFACE':
                self.plot_surface(context, func_material, collection)
            
            end_progress(context)
            self.report({'INFO'}, f"Function plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error plotting function: {e}")
            return {'CANCELLED'}
    
    def plot_curve(self, context, material, collection):
        """Plot a 2D curve y=f(x)"""
        # Generate points
        x_range = np.linspace(self.x_min, self.x_max, self.samples)
        points = []
        
        # Evaluate function at each point
        for i, x in enumerate(x_range):
            # Report progress
            progress = (i + 1) / len(x_range)
            report_progress(context, progress * 0.7, f"Evaluating function: point {i+1}/{len(x_range)}")
            
            try:
                y = evaluate_expression(self.function, {"x": x})
                # Check for valid result
                if not np.isnan(y) and not np.isinf(y):
                    points.append((x, 0, y))
            except ValueError:
                # Skip points where the function is undefined
                continue
        
        if not points:
            raise ValueError("Function evaluation produced no valid points")
        
        report_progress(context, 0.8, "Creating curve object...")
        
        # Create curve
        curve_data = bpy.data.curves.new(f'Function_{self.function}', 'CURVE')
        curve_data.dimensions = '3D'
        
        # Set curve properties
        curve_data.bevel_depth = self.thickness
        curve_data.bevel_resolution = 4
        curve_data.use_fill_caps = True
        
        # Create spline
        spline = curve_data.splines.new('POLY')
        spline.points.add(len(points) - 1)
        
        for i, point in enumerate(points):
            spline.points[i].co = (*point, 1)
        
        # Create curve object
        curve_obj = bpy.data.objects.new(f"Function_{self.function}", curve_data)
        
        # Add to collection
        if curve_obj.users_collection:
            curve_obj.users_collection[0].objects.unlink(curve_obj)
        
        collection.objects.link(curve_obj)
        
        # Apply material
        apply_material(curve_obj, material)
        
        report_progress(context, 0.9, "Creating coordinate axes...")
        
        # Add axes
        self.create_axes()
    
    def plot_surface(self, context, material, collection):
        """Plot a 3D surface z=f(x,y)"""
        # Generate grid
        x = np.linspace(self.x_min, self.x_max, self.samples)
        y = np.linspace(self.y_min, self.y_max, self.samples)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate function at each point
        Z = np.zeros(X.shape)
        
        total_points = X.shape[0] * X.shape[1]
        point_count = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                # Report progress
                point_count += 1
                progress = point_count / total_points
                report_progress(context, progress * 0.7, 
                                f"Evaluating surface: point {point_count}/{total_points}")
                
                try:
                    Z[i, j] = evaluate_expression(self.z_function, {"x": X[i, j], "y": Y[i, j]})
                    # Check for valid result
                    if np.isnan(Z[i, j]) or np.isinf(Z[i, j]):
                        Z[i, j] = 0
                except ValueError:
                    Z[i, j] = 0
        
        report_progress(context, 0.8, "Creating surface mesh...")
        
        # Create mesh
        mesh_data = bpy.data.meshes.new(f"Function_Surface")
        
        # Create vertices
        vertices = []
        faces = []
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                vertices.append((X[i, j], Y[i, j], Z[i, j]))
        
        # Create faces
        for i in range(X.shape[0] - 1):
            for j in range(X.shape[1] - 1):
                base = i * X.shape[1] + j
                faces.append((base, base + 1, base + X.shape[1] + 1, base + X.shape[1]))
        
        # Create the mesh
        mesh_data.from_pydata(vertices, [], faces)
        mesh_data.update()
        
        # Create the object
        surface_obj = bpy.data.objects.new(f"Function_Surface", mesh_data)
        
        # Add to collection
        if surface_obj.users_collection:
            surface_obj.users_collection[0].objects.unlink(surface_obj)
        
        collection.objects.link(surface_obj)
        
        # Apply material
        apply_material(surface_obj, material)
        
        # Set smooth shading
        bpy.context.view_layer.objects.active = surface_obj
        bpy.ops.object.shade_smooth()
        
        report_progress(context, 0.9, "Creating coordinate axes...")
        
        # Add axes
        self.create_axes(is_3d=True)
    
    def create_axes(self, is_3d=False):
        """Create coordinate axes"""
        collection = get_collection("Math_Functions")
        
        # Create x-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=self.x_max - self.x_min,
            enter_editmode=False,
            align='WORLD',
            location=((self.x_max + self.x_min) / 2, 0, 0),
            rotation=(0, math.pi/2, 0),
            scale=(1, 1, 1)
        )
        x_axis = bpy.context.active_object
        x_axis.name = "X_Axis"
        
        # Create x-axis label
        bpy.ops.object.text_add(location=(self.x_max + 0.5, 0, 0))
        x_label = bpy.context.active_object
        x_label.data.body = "X"
        x_label.name = "X_Label"
        
        # Create z-axis (y in mathematical terms)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=abs(self.x_max - self.x_min),
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, (self.x_max - self.x_min) / 2),
            rotation=(0, 0, 0),
            scale=(1, 1, 1)
        )
        z_axis = bpy.context.active_object
        z_axis.name = "Z_Axis"
        
        # Create z-axis label
        bpy.ops.object.text_add(location=(0, 0, self.x_max + 0.5))
        z_label = bpy.context.active_object
        z_label.data.body = "Z"
        z_label.name = "Z_Label"
        
        # Create materials
        x_material = create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        z_material = create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        apply_material(x_axis, x_material)
        apply_material(z_axis, z_material)
        
        # Add to collection
        for obj in [x_axis, x_label, z_axis, z_label]:
            if obj.users_collection:
                obj.users_collection[0].objects.unlink(obj)
            collection.objects.link(obj)
        
        if is_3d:
            # Create y-axis
            bpy.ops.mesh.primitive_cylinder_add(
                radius=0.02,
                depth=self.y_max - self.y_min,
                enter_editmode=False,
                align='WORLD',
                location=(0, (self.y_max + self.y_min) / 2, 0),
                rotation=(math.pi/2, 0, 0),
                scale=(1, 1, 1)
            )
            y_axis = bpy.context.active_object
            y_axis.name = "Y_Axis"
            
            # Create y-axis label
            bpy.ops.object.text_add(location=(0, self.y_max + 0.5, 0))
            y_label = bpy.context.active_object
            y_label.data.body = "Y"
            y_label.name = "Y_Label"
            
            # Create material
            y_material = create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
            
            # Apply material
            apply_material(y_axis, y_material)
            
            # Add to collection
            for obj in [y_axis, y_label]:
                if obj.users_collection:
                    obj.users_collection[0].objects.unlink(obj)
                collection.objects.link(obj)
    
    def invoke(self, context, event):
        # Initialize with current settings
        self.function = context.scene.math_playground.function_expression
        self.x_min = context.scene.math_playground.x_min
        self.x_max = context.scene.math_playground.x_max
        return self.execute(context)

class MATH_OT_PlotParametric(Operator):
    """Plot a parametric curve"""
    bl_idname = "math.plot_parametric"
    bl_label = "Plot Parametric Curve"
    bl_options = {'REGISTER', 'UNDO'}
    
    x_function: StringProperty(
        name="X Function",
        description="Python expression for x=f(t)",
        default="math.cos(t)",
    )
    
    y_function: StringProperty(
        name="Y Function",
        description="Python expression for y=f(t)",
        default="math.sin(t)",
    )
    
    z_function: StringProperty(
        name="Z Function",
        description="Python expression for z=f(t)",
        default="t/10",
    )
    
    t_min: FloatProperty(
        name="T Min",
        description="Minimum t value",
        default=0.0,
    )
    
    t_max: FloatProperty(
        name="T Max",
        description="Maximum t value",
        default=6.28,
    )
    
    samples: IntProperty(
        name="Samples",
        description="Number of sample points",
        default=100,
        min=10,
        max=1000,
    )
    
    thickness: FloatProperty(
        name="Curve Thickness",
        description="Thickness of the curve",
        default=0.05,
        min=0.01,
        max=0.5,
    )
    
    color: FloatVectorProperty(
        name="Color",
        description="Curve color",
        default=(0.8, 0.2, 0.8, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    def execute(self, context):
        # Create function collection if it doesn't exist
        collection = get_collection("Math_Functions")
        
        # Create material for function
        func_material = create_material("Parametric_Material", self.color)
        
        try:
            report_progress(context, 0.0, "Starting parametric curve plot...")
            
            # Generate points
            t_range = np.linspace(self.t_min, self.t_max, self.samples)
            points = []
            
            # Evaluate function at each point
            for i, t in enumerate(t_range):
                # Report progress
                progress = (i + 1) / len(t_range)
                report_progress(context, progress * 0.7, 
                                f"Evaluating curve: point {i+1}/{len(t_range)}")
                
                try:
                    x = evaluate_expression(self.x_function, {"t": t})
                    y = evaluate_expression(self.y_function, {"t": t})
                    z = evaluate_expression(self.z_function, {"t": t})
                    
                    # Check for valid results
                    if (not np.isnan(x) and not np.isinf(x) and
                        not np.isnan(y) and not np.isinf(y) and
                        not np.isnan(z) and not np.isinf(z)):
                        points.append((x, y, z))
                except ValueError:
                    # Skip points where the function is undefined
                    continue
            
            if not points:
                raise ValueError("Function evaluation produced no valid points")
            
            report_progress(context, 0.8, "Creating curve object...")
            
            # Create curve
            curve_data = bpy.data.curves.new('Parametric_Curve', 'CURVE')
            curve_data.dimensions = '3D'
            
            # Set curve properties
            curve_data.bevel_depth = self.thickness
            curve_data.bevel_resolution = 4
            curve_data.use_fill_caps = True
            
            # Create spline
            spline = curve_data.splines.new('POLY')
            spline.points.add(len(points) - 1)
            
            for i, point in enumerate(points):
                spline.points[i].co = (*point, 1)
            
            # Create curve object
            curve_obj = bpy.data.objects.new("Parametric_Curve", curve_data)
            
            # Add to collection
            if curve_obj.users_collection:
                curve_obj.users_collection[0].objects.unlink(curve_obj)
            
            collection.objects.link(curve_obj)
            
            # Apply material
            apply_material(curve_obj, func_material)
            
            report_progress(context, 0.9, "Creating coordinate axes...")
            
            # Add axes
            self.create_axes()
            
            end_progress(context)
            self.report({'INFO'}, "Parametric curve plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error plotting parametric curve: {e}")
            return {'CANCELLED'}
    
    def create_axes(self):
        """Create coordinate axes"""
        collection = get_collection("Math_Functions")
        
        # Calculate axis lengths based on the curve extents
        axis_length = 10  # Default length
        
        # Create x-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length * 2,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            rotation=(0, math.pi/2, 0),
            scale=(1, 1, 1)
        )
        x_axis = bpy.context.active_object
        x_axis.name = "X_Axis"
        
        # Create x-axis label
        bpy.ops.object.text_add(location=(axis_length, 0, 0))
        x_label = bpy.context.active_object
        x_label.data.body = "X"
        x_label.name = "X_Label"
        
        # Create y-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length * 2,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            rotation=(math.pi/2, 0, 0),
            scale=(1, 1, 1)
        )
        y_axis = bpy.context.active_object
        y_axis.name = "Y_Axis"
        
        # Create y-axis label
        bpy.ops.object.text_add(location=(0, axis_length, 0))
        y_label = bpy.context.active_object
        y_label.data.body = "Y"
        y_label.name = "Y_Label"
        
        # Create z-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length * 2,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            rotation=(0, 0, 0),
            scale=(1, 1, 1)
        )
        z_axis = bpy.context.active_object
        z_axis.name = "Z_Axis"
        
        # Create z-axis label
        bpy.ops.object.text_add(location=(0, 0, axis_length))
        z_label = bpy.context.active_object
        z_label.data.body = "Z"
        z_label.name = "Z_Label"
        
        # Create materials
        x_material = create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        y_material = create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
        z_material = create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        apply_material(x_axis, x_material)
        apply_material(y_axis, y_material)
        apply_material(z_axis, z_material)
        
        # Add to collection
        for obj in [x_axis, x_label, y_axis, y_label, z_axis, z_label]:
            if obj.users_collection:
                obj.users_collection[0].objects.unlink(obj)
            collection.objects.link(obj)

class MATH_OT_PlotVectorField(Operator):
    """Plot a vector field"""
    bl_idname = "math.plot_vector_field"
    bl_label = "Plot Vector Field"
    bl_options = {'REGISTER', 'UNDO'}
    
    x_component: StringProperty(
        name="X Component",
        description="Python expression for x component (use x, y, z)",
        default="-y",
    )
    
    y_component: StringProperty(
        name="Y Component",
        description="Python expression for y component (use x, y, z)",
        default="x",
    )
    
    z_component: StringProperty(
        name="Z Component",
        description="Python expression for z component (use x, y, z)",
        default="0",
    )
    
    x_min: FloatProperty(
        name="X Min",
        description="Minimum x value",
        default=-5.0,
    )
    
    x_max: FloatProperty(
        name="X Max",
        description="Maximum x value",
        default=5.0,
    )
    
    y_min: FloatProperty(
        name="Y Min",
        description="Minimum y value",
        default=-5.0,
    )
    
    y_max: FloatProperty(
        name="Y Max",
        description="Maximum y value",
        default=5.0,
    )
    
    z_min: FloatProperty(
        name="Z Min",
        description="Minimum z value",
        default=-5.0,
    )
    
    z_max: FloatProperty(
        name="Z Max",
        description="Maximum z value",
        default=5.0,
    )
    
    grid_size: IntProperty(
        name="Grid Size",
        description="Number of grid points in each dimension",
        default=5,
        min=2,
        max=20,
    )
    
    arrow_scale: FloatProperty(
        name="Arrow Scale",
        description="Scale factor for arrows",
        default=0.5,
        min=0.1,
        max=5.0,
    )
    
    normalize: BoolProperty(
        name="Normalize Vectors",
        description="Normalize vector lengths",
        default=False,
    )
    
    color: FloatVectorProperty(
        name="Color",
        description="Vector field color",
        default=(0.8, 0.4, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    def execute(self, context):
        # Create vector field collection if it doesn't exist
        collection = get_collection("Math_VectorField")
        
        # Clear existing vector field
        clear_collection("Math_VectorField")
        
        # Create material for vector field
        field_material = create_material("VectorField_Material", self.color)
        
        try:
            report_progress(context, 0.0, "Starting vector field plot...")
            
            # Generate grid points
            x_range = np.linspace(self.x_min, self.x_max, self.grid_size)
            y_range = np.linspace(self.y_min, self.y_max, self.grid_size)
            z_range = np.linspace(self.z_min, self.z_max, self.grid_size)
            
            # Calculate total number of grid points
            total_points = self.grid_size**3
            point_count = 0
            
            # Create arrows at each grid point
            for x in x_range:
                for y in y_range:
                    for z in z_range:
                        # Report progress
                        point_count += 1
                        progress = point_count / total_points
                        report_progress(context, progress * 0.8, 
                                       f"Creating vector {point_count}/{total_points}")
                        
                        # Evaluate vector components
                        try:
                            vx = evaluate_expression(self.x_component, {"x": x, "y": y, "z": z})
                            vy = evaluate_expression(self.y_component, {"x": x, "y": y, "z": z})
                            vz = evaluate_expression(self.z_component, {"x": x, "y": y, "z": z})
                            
                            # Create a vector
                            vec = Vector((vx, vy, vz))
                            
                            # Normalize if requested
                            if self.normalize and vec.length > 0:
                                vec.normalize()
                            
                            # Scale the vector
                            vec = vec * self.arrow_scale
                            
                            # Create an arrow
                            if vec.length > 0.01:  # Only create visible arrows
                                self.create_arrow((x, y, z), vec, field_material, collection)
                        
                        except ValueError as e:
                            # Skip points where the function is undefined
                            print(f"Error at point ({x}, {y}, {z}): {e}")
                            continue
            
            report_progress(context, 0.9, "Creating coordinate axes...")
            
            # Add axes
            self.create_axes()
            
            end_progress(context)
            self.report({'INFO'}, "Vector field plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error plotting vector field: {e}")
            return {'CANCELLED'}
    
    def create_arrow(self, start, vector, material, collection):
        """Create an arrow representing a vector at the given start point"""
        # Create arrow shaft (cylinder)
        shaft_depth = vector.length * 0.8
        shaft_radius = shaft_depth * 0.05
        
        bpy.ops.mesh.primitive_cylinder_add(
            radius=shaft_radius,
            depth=shaft_depth,
            enter_editmode=False,
            align='WORLD',
            location=start,
            scale=(1, 1, 1)
        )
        shaft = bpy.context.active_object
        
        # Create arrow head (cone)
        head_depth = vector.length * 0.2
        head_radius = shaft_radius * 2
        
        bpy.ops.mesh.primitive_cone_add(
            radius1=head_radius,
            radius2=0,
            depth=head_depth,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            scale=(1, 1, 1)
        )
        head = bpy.context.active_object
        
        # Position arrow head at the end of the shaft
        head.location = (
            start[0] + vector[0] * 0.9,
            start[1] + vector[1] * 0.9,
            start[2] + vector[2] * 0.9
        )
        
        # Align arrow to vector direction
        if vector.length > 0:
            # Create a rotation from the Z axis to the vector direction
            z_axis = Vector((0, 0, 1))
            vec_norm = vector.normalized()
            
            rotation_axis = z_axis.cross(vec_norm)
            
            if rotation_axis.length > 0:
                rotation_angle = math.acos(min(1, max(-1, z_axis.dot(vec_norm))))
                rotation_axis.normalize()
                rotation = rotation_axis.to_track_quat('Z', 'Y').to_euler()
                
                shaft.rotation_euler = rotation
                head.rotation_euler = rotation
                
                # Move shaft to correct position
                shaft.location = (
                    start[0] + vector[0] * 0.4,
                    start[1] + vector[1] * 0.4,
                    start[2] + vector[2] * 0.4
                )
        
        # Apply material
        apply_material(shaft, material)
        apply_material(head, material)
        
        # Add to collection
        if shaft.users_collection:
            shaft.users_collection[0].objects.unlink(shaft)
        if head.users_collection:
            head.users_collection[0].objects.unlink(head)
        
        collection.objects.link(shaft)
        collection.objects.link(head)
    
    def create_axes(self):
        """Create coordinate axes"""
        collection = get_collection("Math_VectorField")
        
        axis_length = max(
            abs(self.x_max - self.x_min),
            abs(self.y_max - self.y_min),
            abs(self.z_max - self.z_min)
        )
        
        # Create x-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length,
            enter_editmode=False,
            align='WORLD',
            location=(axis_length/2, 0, 0),
            rotation=(0, math.pi/2, 0),
            scale=(1, 1, 1)
        )
        x_axis = bpy.context.active_object
        x_axis.name = "X_Axis"
        
        # Create x-axis label
        bpy.ops.object.text_add(location=(axis_length, 0, 0))
        x_label = bpy.context.active_object
        x_label.data.body = "X"
        x_label.name = "X_Label"
        
        # Create y-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length,
            enter_editmode=False,
            align='WORLD',
            location=(0, axis_length/2, 0),
            rotation=(math.pi/2, 0, 0),
            scale=(1, 1, 1)
        )
        y_axis = bpy.context.active_object
        y_axis.name = "Y_Axis"
        
        # Create y-axis label
        bpy.ops.object.text_add(location=(0, axis_length, 0))
        y_label = bpy.context.active_object
        y_label.data.body = "Y"
        y_label.name = "Y_Label"
        
        # Create z-axis
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.02,
            depth=axis_length,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, axis_length/2),
            rotation=(0, 0, 0),
            scale=(1, 1, 1)
        )
        z_axis = bpy.context.active_object
        z_axis.name = "Z_Axis"
        
        # Create z-axis label
        bpy.ops.object.text_add(location=(0, 0, axis_length))
        z_label = bpy.context.active_object
        z_label.data.body = "Z"
        z_label.name = "Z_Label"
        
        # Create materials
        x_material = create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        y_material = create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
        z_material = create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        apply_material(x_axis, x_material)
        apply_material(y_axis, y_material)
        apply_material(z_axis, z_material)
        
        # Add to collection
        for obj in [x_axis, x_label, y_axis, y_label, z_axis, z_label]:
            if obj.users_collection:
                obj.users_collection[0].objects.unlink(obj)
            collection.objects.link(obj)

class MATH_OT_ClearAnalysis(Operator):
    """Clear all analysis objects from the scene"""
    bl_idname = "math.clear_analysis"
    bl_label = "Clear Analysis"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Functions")
        clear_collection("Math_VectorField")
        self.report({'INFO'}, "All analysis objects cleared")
        return {'FINISHED'}

# ------------------------------------------------
# Graph Theory Module
# ------------------------------------------------

class MATH_OT_CreateGraph(Operator):
    """Create a graph with nodes and edges"""
    bl_idname = "math.create_graph"
    bl_label = "Create Graph"
    bl_options = {'REGISTER', 'UNDO'}
    
    num_nodes: IntProperty(
        name="Number of Nodes",
        description="Number of nodes in the graph",
        default=10,
        min=2,
        max=100,
    )
    
    graph_type: EnumProperty(
        name="Graph Type",
        description="Type of graph to create",
        items=[
            ('CIRCLE', "Circle", "Arrange nodes in a circle"),
            ('RANDOM_2D', "Random 2D", "Arrange nodes randomly in 2D"),
            ('RANDOM_3D', "Random 3D", "Arrange nodes randomly in 3D"),
            ('GRID_2D', "Grid 2D", "Arrange nodes in a 2D grid"),
            ('CUBE_3D', "Cube 3D", "Arrange nodes in a 3D cube")
        ],
        default='CIRCLE',
    )
    
    connection_type: EnumProperty(
        name="Connection Type",
        description="How to connect nodes",
        items=[
            ('RANDOM', "Random", "Connect nodes randomly"),
            ('NEAREST', "Nearest Neighbors", "Connect each node to its nearest neighbors"),
            ('COMPLETE', "Complete", "Connect all nodes to each other"),
            ('TREE', "Tree", "Connect nodes as a tree"),
            ('CUSTOM', "Custom", "User-defined edge list")
        ],
        default='RANDOM',
    )
    
    edge_probability: FloatProperty(
        name="Edge Probability",
        description="Probability of creating an edge between nodes (for random connections)",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    
    num_neighbors: IntProperty(
        name="Number of Neighbors",
        description="Number of nearest neighbors to connect (for nearest neighbor connections)",
        default=2,
        min=1,
        max=20,
    )
    
    custom_edges: StringProperty(
        name="Custom Edges",
        description="Custom edge list format: '0-1,1-2,2-3' (for custom connections)",
        default="0-1,1-2,2-0",
    )
    
    node_size: FloatProperty(
        name="Node Size",
        description="Size of the graph nodes",
        default=0.2,
        min=0.05,
        max=1.0,
    )
    
    edge_thickness: FloatProperty(
        name="Edge Thickness",
        description="Thickness of the graph edges",
        default=0.05,
        min=0.01,
        max=0.2,
    )
    
    node_color: FloatVectorProperty(
        name="Node Color",
        description="Color of the graph nodes",
        default=(0.2, 0.6, 0.9, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    edge_color: FloatVectorProperty(
        name="Edge Color",
        description="Color of the graph edges",
        default=(0.6, 0.6, 0.6, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    def execute(self, context):
        try:
            # Create graph collection if it doesn't exist
            collection = get_collection("Math_Graphs")
            
            # Clear existing graph
            clear_collection("Math_Graphs")
            
            report_progress(context, 0.0, "Starting graph creation...")
            
            # Create materials
            node_material = create_material("Graph_Node_Material", self.node_color)
            edge_material = create_material("Graph_Edge_Material", self.edge_color)
            
            # Generate node positions
            report_progress(context, 0.1, "Generating node positions...")
            node_positions = self.generate_node_positions()
            
            # Generate edges
            report_progress(context, 0.2, "Generating edges...")
            edges = self.generate_edges(node_positions)
            
            # Create nodes
            report_progress(context, 0.3, "Creating nodes...")
            nodes = self.create_nodes(context, node_positions, node_material, collection)
            
            # Create edges
            report_progress(context, 0.7, "Creating edges...")
            self.create_edges(context, edges, node_positions, edge_material, collection)
            
            end_progress(context)
            self.report({'INFO'}, f"Created graph with {len(node_positions)} nodes and {len(edges)} edges")
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error creating graph: {e}")
            return {'CANCELLED'}
    
    def generate_node_positions(self):
        """Generate positions for graph nodes based on the selected layout"""
        positions = []
        
        if self.graph_type == 'CIRCLE':
            # Arrange nodes in a circle
            radius = self.num_nodes * 0.2
            for i in range(self.num_nodes):
                angle = 2 * math.pi * i / self.num_nodes
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 0
                positions.append((x, y, z))
        
        elif self.graph_type == 'RANDOM_2D':
            # Arrange nodes randomly in 2D
            scale = self.num_nodes * 0.3
            for _ in range(self.num_nodes):
                x = random.uniform(-scale, scale)
                y = random.uniform(-scale, scale)
                z = 0
                positions.append((x, y, z))
        
        elif self.graph_type == 'RANDOM_3D':
            # Arrange nodes randomly in 3D
            scale = self.num_nodes * 0.3
            for _ in range(self.num_nodes):
                x = random.uniform(-scale, scale)
                y = random.uniform(-scale, scale)
                z = random.uniform(-scale, scale)
                positions.append((x, y, z))
        
        elif self.graph_type == 'GRID_2D':
            # Arrange nodes in a 2D grid
            side = math.ceil(math.sqrt(self.num_nodes))
            spacing = 2.0
            for i in range(self.num_nodes):
                row = i // side
                col = i % side
                x = col * spacing - (side - 1) * spacing / 2
                y = row * spacing - (side - 1) * spacing / 2
                z = 0
                positions.append((x, y, z))
        
        elif self.graph_type == 'CUBE_3D':
            # Arrange nodes in a 3D cube
            side = math.ceil(self.num_nodes ** (1/3))
            spacing = 2.0
            for i in range(self.num_nodes):
                z_layer = i // (side * side)
                remaining = i % (side * side)
                row = remaining // side
                col = remaining % side
                x = col * spacing - (side - 1) * spacing / 2
                y = row * spacing - (side - 1) * spacing / 2
                z = z_layer * spacing - (side - 1) * spacing / 2
                positions.append((x, y, z))
        
        return positions
    
    def generate_edges(self, node_positions):
        """Generate edges between nodes based on the selected connection type"""
        edges = []
        
        if self.connection_type == 'RANDOM':
            # Connect nodes randomly with given probability
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    if random.random() < self.edge_probability:
                        edges.append((i, j))
        
        elif self.connection_type == 'NEAREST':
            # Connect each node to its nearest neighbors
            for i in range(self.num_nodes):
                # Calculate distances to all other nodes
                distances = []
                for j in range(self.num_nodes):
                    if i != j:
                        dist = math.sqrt(
                            (node_positions[i][0] - node_positions[j][0]) ** 2 +
                            (node_positions[i][1] - node_positions[j][1]) ** 2 +
                            (node_positions[i][2] - node_positions[j][2]) ** 2
                        )
                        distances.append((j, dist))
                
                # Sort by distance
                distances.sort(key=lambda x: x[1])
                
                # Connect to nearest neighbors
                for j, _ in distances[:self.num_neighbors]:
                    if i < j:  # Avoid duplicates
                        edges.append((i, j))
                    else:
                        edges.append((j, i))
            
            # Remove duplicate edges
            edges = list(set(edges))
        
        elif self.connection_type == 'COMPLETE':
            # Connect all nodes to each other
            for i in range(self.num_nodes):
                for j in range(i + 1, self.num_nodes):
                    edges.append((i, j))
        
        elif self.connection_type == 'TREE':
            # Create a minimum spanning tree (simplified approach)
            # Start with a single node and add edges to the closest unconnected node
            connected = {0}  # Start with node 0
            while len(connected) < self.num_nodes:
                min_dist = float('inf')
                best_edge = None
                
                # Find the closest unconnected node to any connected node
                for i in connected:
                    for j in range(self.num_nodes):
                        if j not in connected:
                            dist = math.sqrt(
                                (node_positions[i][0] - node_positions[j][0]) ** 2 +
                                (node_positions[i][1] - node_positions[j][1]) ** 2 +
                                (node_positions[i][2] - node_positions[j][2]) ** 2
                            )
                            if dist < min_dist:
                                min_dist = dist
                                best_edge = (i, j)
                
                if best_edge:
                    edges.append(best_edge)
                    connected.add(best_edge[1])
        
        elif self.connection_type == 'CUSTOM':
            # Parse custom edge list
            try:
                edge_list = self.custom_edges.split(',')
                for edge_str in edge_list:
                    if '-' in edge_str:
                        start, end = edge_str.split('-')
                        start_idx = int(start.strip())
                        end_idx = int(end.strip())
                        if 0 <= start_idx < self.num_nodes and 0 <= end_idx < self.num_nodes:
                            edges.append((start_idx, end_idx))
            except Exception as e:
                raise ValueError(f"Error parsing custom edges: {e}")
        
        return edges
    
    def create_nodes(self, context, node_positions, material, collection):
        """Create nodes for the graph"""
        nodes = []
        
        for i, pos in enumerate(node_positions):
            # Report progress
            progress = 0.3 + (i / len(node_positions)) * 0.4
            report_progress(context, progress, f"Creating node {i+1}/{len(node_positions)}")
            
            # Create node (sphere)
            bpy.ops.mesh.primitive_uv_sphere_add(
                radius=self.node_size,
                location=pos,
                segments=16,
                ring_count=8
            )
            node = bpy.context.active_object
            node.name = f"Node_{i}"
            
            # Add text to show node index
            bpy.ops.object.text_add(location=(pos[0], pos[1], pos[2] + self.node_size + 0.1))
            text = bpy.context.active_object
            text.data.body = str(i)
            text.data.align_x = 'CENTER'
            text.rotation_euler.x = math.pi / 2
            text.name = f"NodeText_{i}"
            
            # Apply material
            apply_material(node, material)
            
            # Add to collection
            if node.users_collection:
                node.users_collection[0].objects.unlink(node)
            if text.users_collection:
                text.users_collection[0].objects.unlink(text)
            
            collection.objects.link(node)
            collection.objects.link(text)
            
            nodes.append(node)
        
        return nodes
    
    def create_edges(self, context, edges, node_positions, material, collection):
        """Create edges between nodes"""
        for i, edge in enumerate(edges):
            # Report progress
            progress = 0.7 + (i / len(edges)) * 0.3
            report_progress(context, progress, f"Creating edge {i+1}/{len(edges)}")
            
            start_pos = node_positions[edge[0]]
            end_pos = node_positions[edge[1]]
            
            # Calculate edge properties
            edge_vector = Vector((
                end_pos[0] - start_pos[0],
                end_pos[1] - start_pos[1],
                end_pos[2] - start_pos[2]
            ))
            edge_length = edge_vector.length
            edge_center = (
                (start_pos[0] + end_pos[0]) / 2,
                (start_pos[1] + end_pos[1]) / 2,
                (start_pos[2] + end_pos[2]) / 2
            )
            
            # Create edge (cylinder)
            bpy.ops.mesh.primitive_cylinder_add(
                radius=self.edge_thickness,
                depth=edge_length,
                location=edge_center,
                vertices=8
            )
            edge_obj = bpy.context.active_object
            edge_obj.name = f"Edge_{edge[0]}_{edge[1]}"
            
            # Align cylinder to edge direction
            if edge_length > 0:
                # Create a rotation from the Z axis to the edge direction
                z_axis = Vector((0, 0, 1))
                edge_dir = edge_vector.normalized()
                
                rotation_axis = z_axis.cross(edge_dir)
                
                if rotation_axis.length > 0:
                    rotation_angle = math.acos(min(1, max(-1, z_axis.dot(edge_dir))))
                    rotation_axis.normalize()
                    edge_obj.rotation_euler = rotation_axis.to_track_quat('Z', 'Y').to_euler()
            
            # Apply material
            apply_material(edge_obj, material)
            
            # Add to collection
            if edge_obj.users_collection:
                edge_obj.users_collection[0].objects.unlink(edge_obj)
            
            collection.objects.link(edge_obj)
    
    def invoke(self, context, event):
        # Initialize with current settings
        self.num_nodes = context.scene.math_playground.num_nodes
        self.edge_probability = context.scene.math_playground.edge_probability
        return self.execute(context)

class MATH_OT_RunGraphAlgorithm(Operator):
    """Run a graph algorithm on the current graph"""
    bl_idname = "math.run_graph_algorithm"
    bl_label = "Run Graph Algorithm"
    bl_options = {'REGISTER', 'UNDO'}
    
    algorithm: EnumProperty(
        name="Algorithm",
        description="Graph algorithm to run",
        items=[
            ('SHORTEST_PATH', "Shortest Path", "Find shortest path between two nodes"),
            ('MIN_SPANNING_TREE', "Minimum Spanning Tree", "Find minimum spanning tree"),
            ('GRAPH_COLORING', "Graph Coloring", "Color the graph with minimum colors")
        ],
        default='SHORTEST_PATH',
    )
    
    start_node: IntProperty(
        name="Start Node",
        description="Starting node for shortest path algorithm",
        default=0,
        min=0,
    )
    
    end_node: IntProperty(
        name="End Node",
        description="Ending node for shortest path algorithm",
        default=1,
        min=0,
    )
    
    def execute(self, context):
        try:
            # Get graph collection
            collection = bpy.data.collections.get("Math_Graphs")
            if not collection:
                self.report({'ERROR'}, "No graph found. Create a graph first.")
                return {'CANCELLED'}
            
            report_progress(context, 0.0, "Extracting graph structure...")
            
            # Extract graph structure
            nodes, edges = self.extract_graph_structure(collection)
            
            # Check if nodes exist
            if not nodes:
                self.report({'ERROR'}, "No nodes found in the graph.")
                end_progress(context)
                return {'CANCELLED'}
            
            # Validate start and end nodes
            if self.start_node >= len(nodes) or self.end_node >= len(nodes):
                self.report({'ERROR'}, "Invalid start or end node index.")
                end_progress(context)
                return {'CANCELLED'}
            
            report_progress(context, 0.2, f"Running {self.algorithm} algorithm...")
            
            # Run selected algorithm
            if self.algorithm == 'SHORTEST_PATH':
                self.run_shortest_path(context, nodes, edges)
            
            elif self.algorithm == 'MIN_SPANNING_TREE':
                self.run_minimum_spanning_tree(context, nodes, edges)
            
            elif self.algorithm == 'GRAPH_COLORING':
                self.run_graph_coloring(context, nodes, edges)
            
            end_progress(context)
            return {'FINISHED'}
            
        except Exception as e:
            end_progress(context)
            self.report({'ERROR'}, f"Error running algorithm: {e}")
            return {'CANCELLED'}
    
    def extract_graph_structure(self, collection):
        """Extract graph structure from collection objects"""
        nodes = []
        edges = []
        
        # Find the maximum node index
        max_node_idx = -1
        for obj in collection.objects:
            if obj.name.startswith("Node_"):
                try:
                    idx = int(obj.name.split("_")[1])
                    max_node_idx = max(max_node_idx, idx)
                except ValueError:
                    continue
        
        # Initialize nodes list with None placeholders
        nodes = [None] * (max_node_idx + 1)
        
        # Get node objects
        for obj in collection.objects:
            if obj.name.startswith("Node_"):
                try:
                    idx = int(obj.name.split("_")[1])
                    nodes[idx] = obj
                except ValueError:
                    continue
        
        # Get edge objects and their connections
        for obj in collection.objects:
            if obj.name.startswith("Edge_"):
                parts = obj.name.split("_")
                if len(parts) >= 3:
                    try:
                        start = int(parts[1])
                        end = int(parts[2])
                        edges.append((start, end, obj))
                    except ValueError:
                        continue
        
        return nodes, edges
    
    def run_shortest_path(self, context, nodes, edges):
        """Run Dijkstra's shortest path algorithm"""
        report_progress(context, 0.3, "Building adjacency list...")
        
        # Build adjacency list
        adj_list = [[] for _ in range(len(nodes))]
        for start, end, _ in edges:
            # Calculate distance (weight) between nodes
            start_pos = nodes[start].location
            end_pos = nodes[end].location
            distance = math.sqrt(
                (start_pos[0] - end_pos[0]) ** 2 +
                (start_pos[1] - end_pos[1]) ** 2 +
                (start_pos[2] - end_pos[2]) ** 2
            )
            
            # Add edges in both directions (undirected graph)
            adj_list[start].append((end, distance))
            adj_list[end].append((start, distance))
        
        report_progress(context, 0.4, "Running Dijkstra's algorithm...")
        
        # Initialize Dijkstra's algorithm
        distances = [float('inf')] * len(nodes)
        distances[self.start_node] = 0
        previous = [None] * len(nodes)
        unvisited = set(range(len(nodes)))
        
        while unvisited:
            # Find the unvisited node with the smallest distance
            current = min(unvisited, key=lambda x: distances[x])
            
            # If we reached the end node or if the smallest distance is infinity, stop
            if current == self.end_node or distances[current] == float('inf'):
                break
            
            # Remove the current node from unvisited
            unvisited.remove(current)
            
            # Update distances to neighbors
            for neighbor, weight in adj_list[current]:
                if neighbor in unvisited:
                    distance = distances[current] + weight
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
        
        # Reconstruct the path
        if distances[self.end_node] == float('inf'):
            self.report({'INFO'}, "No path found between the selected nodes.")
            return
        
        path = []
        current = self.end_node
        while current is not None:
            path.append(current)
            current = previous[current]
        
        path.reverse()
        
        report_progress(context, 0.7, "Highlighting path...")
        
        # Highlight the path
        self.highlight_path(nodes, edges, path)
        
        self.report({'INFO'}, f"Shortest path found: {' -> '.join(map(str, path))}")
    
    def run_minimum_spanning_tree(self, context, nodes, edges):
        """Run Kruskal's minimum spanning tree algorithm"""
        report_progress(context, 0.3, "Building edge list...")
        
        # Build edge list with weights
        weighted_edges = []
        for start, end, edge_obj in edges:
            # Calculate distance (weight) between nodes
            start_pos = nodes[start].location
            end_pos = nodes[end].location
            distance = math.sqrt(
                (start_pos[0] - end_pos[0]) ** 2 +
                (start_pos[1] - end_pos[1]) ** 2 +
                (start_pos[2] - end_pos[2]) ** 2
            )
            
            weighted_edges.append((start, end, distance, edge_obj))
        
        report_progress(context, 0.4, "Sorting edges by weight...")
        
        # Sort edges by weight
        weighted_edges.sort(key=lambda x: x[2])
        
        # Initialize disjoint-set data structure
        parent = list(range(len(nodes)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            parent[find(x)] = find(y)
        
        report_progress(context, 0.5, "Running Kruskal's algorithm...")
        
        # Run Kruskal's algorithm
        mst_edges = []
        for start, end, weight, edge_obj in weighted_edges:
            if find(start) != find(end):
                union(start, end)
                mst_edges.append((start, end, edge_obj))
        
        report_progress(context, 0.7, "Highlighting MST...")
        
        # Highlight the MST
        self.highlight_mst(nodes, edges, mst_edges)
        
        self.report({'INFO'}, f"Minimum spanning tree found with {len(mst_edges)} edges.")
    
    def run_graph_coloring(self, context, nodes, edges):
        """Run a greedy graph coloring algorithm"""
        report_progress(context, 0.3, "Building adjacency list...")
        
        # Build adjacency list
        adj_list = [[] for _ in range(len(nodes))]
        for start, end, _ in edges:
            adj_list[start].append(end)
            adj_list[end].append(start)
        
        report_progress(context, 0.4, "Running graph coloring algorithm...")
        
        # Initialize colors
        colors = [-1] * len(nodes)
        
        # Define a set of distinct colors
        color_palette = [
            (1.0, 0.0, 0.0, 1.0),  # Red
            (0.0, 1.0, 0.0, 1.0),  # Green
            (0.0, 0.0, 1.0, 1.0),  # Blue
            (1.0, 1.0, 0.0, 1.0),  # Yellow
            (1.0, 0.0, 1.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0, 1.0),  # Orange
            (0.5, 0.0, 1.0, 1.0),  # Purple
            (0.0, 0.5, 0.0, 1.0),  # Dark Green
            (0.0, 0.0, 0.5, 1.0),  # Dark Blue
        ]
        
        report_progress(context, 0.5, "Coloring nodes...")
        
        # Greedy coloring algorithm
        for node in range(len(nodes)):
            # Find the first available color
            used_colors = set(colors[neighbor] for neighbor in adj_list[node] if colors[neighbor] != -1)
            
            # Find the first available color
            color = 0
            while color in used_colors:
                color += 1
            
            colors[node] = color
        
        report_progress(context, 0.7, "Applying colors to nodes...")
        
        # Apply colors to nodes
        for node_idx, color_idx in enumerate(colors):
            if node_idx < len(nodes) and nodes[node_idx]:
                # Get or create material
                color = color_palette[color_idx % len(color_palette)]
                material_name = f"Node_Color_{color_idx}"
                material = create_material(material_name, color)
                
                # Apply material to node
                apply_material(nodes[node_idx], material)
        
        self.report({'INFO'}, f"Graph colored using {max(colors) + 1} colors.")
    
    def highlight_path(self, nodes, edges, path):
        """Highlight the shortest path"""
        # Create path material
        path_material = create_material("Shortest_Path_Material", (1.0, 0.8, 0.0, 1.0))
        node_highlight_material = create_material("Path_Node_Material", (1.0, 0.4, 0.0, 1.0))
        
        # Highlight nodes in the path
        for node_idx in path:
            if node_idx < len(nodes) and nodes[node_idx]:
                apply_material(nodes[node_idx], node_highlight_material)
        
        # Highlight edges in the path
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i + 1]
            
            # Find the edge object
            for edge_start, edge_end, edge_obj in edges:
                if (edge_start == start and edge_end == end) or (edge_start == end and edge_end == start):
                    apply_material(edge_obj, path_material)
    
    def highlight_mst(self, nodes, edges, mst_edges):
        """Highlight the minimum spanning tree"""
        # Create MST material
        mst_material = create_material("MST_Material", (0.0, 1.0, 0.5, 1.0))
        
        # Reset all edges to the original color
        edge_material = create_material("Graph_Edge_Material", (0.6, 0.6, 0.6, 1.0))
        for _, _, edge_obj in edges:
            apply_material(edge_obj, edge_material)
        
        # Highlight MST edges
        for _, _, edge_obj in mst_edges:
            apply_material(edge_obj, mst_material)

class MATH_OT_ClearGraphs(Operator):
    """Clear all graph objects from the scene"""
    bl_idname = "math.clear_graphs"
    bl_label = "Clear Graphs"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Graphs")
        self.report({'INFO'}, "All graph objects cleared")
        return {'FINISHED'}