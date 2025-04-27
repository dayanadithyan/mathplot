# ui/panels.py - UI panel definitions

import bpy
from bpy.types import Panel

class MATH_PT_Main(Panel):
    """Main panel for Math Playground"""
    bl_label = "Math Playground"
    bl_idname = "MATH_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Math Playground'
    
    def draw(self, context):
            """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
    layout = self.layout
        props = context.scene.math_playground
        
        layout.label(text="Choose a module:")
        
        # Create dropdown for different modules
        layout.prop(props, "active_module", text="")
        
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
            """poll function.
    """
        """poll function.
    """
        """poll function.
    """
        """poll function.
    """
    return context.scene.math_playground.active_module == 'LINEAR_ALGEBRA'
    
    def draw(self, context):
            """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
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
        col.label(text="Format: a,b,c;d,e,f;g,h,i")
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
    
    @classmethod
    def poll(cls, context):
            """poll function.
    """
        """poll function.
    """
        """poll function.
    """
        """poll function.
    """
    return context.scene.math_playground.active_module == 'NUMBER_THEORY'
    
    def draw(self, context):
            """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
    layout = self.layout
        props = context.scene.math_playground.number_theory
        
        # Prime generation
        box = layout.box()
        box.label(text="Generate Primes")
        col = box.column(align=True)
        col.prop(props, "prime_limit", text="Limit")
        op = col.operator("math.generate_primes", text="Generate Primes")
        
        # Sequence generation
        box = layout.box()
        box.label(text="Generate Sequence")
        col = box.column(align=True)
        col.prop(props, "sequence_type", text="Type")
        col.prop(props, "sequence_length", text="Length")
        
        if props.sequence_type == 'CUSTOM':
            col.prop(props, "custom_formula", text="Formula")
            col.label(text="Use 'n' as the term index")
        
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
    
    @classmethod
    def poll(cls, context):
            """poll function.
    """
        """poll function.
    """
        """poll function.
    """
        """poll function.
    """
    return context.scene.math_playground.active_module == 'ANALYSIS'
    
    def draw(self, context):
            """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
    layout = self.layout
        props = context.scene.math_playground.analysis
        
        # Function plotting
        box = layout.box()
        box.label(text="Plot Function")
        col = box.column(align=True)
        col.prop(props, "function_expression", text="f(x)")
        col.prop(props, "x_min", text="X Min")
        col.prop(props, "x_max", text="X Max")
        col.prop(props, "samples", text="Samples")
        op = col.operator("math.plot_function", text="Plot Function")
        
        # Vector field
        box = layout.box()
        box.label(text="Vector Field")
        col = box.column(align=True)
        col.prop(props, "vector_field_x", text="X Component")
        col.prop(props, "vector_field_y", text="Y Component")
        col.prop(props, "vector_field_z", text="Z Component")
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
    
    @classmethod
    def poll(cls, context):
            """poll function.
    """
        """poll function.
    """
        """poll function.
    """
        """poll function.
    """
    return context.scene.math_playground.active_module == 'GRAPH_THEORY'
    
    def draw(self, context):
            """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
        """draw function.
    """
    layout = self.layout
        props = context.scene.math_playground.graph_theory
        
        # Graph creation
        box = layout.box()
        box.label(text="Create Graph")
        col = box.column(align=True)
        col.prop(props, "node_count", text="Nodes")
        col.prop(props, "edge_probability", text="Edge Probability")
        col.prop(props, "layout_type", text="Layout")
        op = col.operator("math.create_graph", text="Create Graph")
        
        # Clear button
        layout.operator("math.clear_graph_theory", text="Clear Graph Theory")

# Registration functions
classes = [
    MATH_PT_Main,
    MATH_PT_LinearAlgebra,
    MATH_PT_NumberTheory,
    MATH_PT_Analysis,
    MATH_PT_GraphTheory,
]

def register():
    """Register UI panels"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister UI panels"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)