# properties.py - Property definitions for Math Playground

import bpy
from bpy.props import (
    StringProperty, BoolProperty, IntProperty, FloatProperty,
    FloatVectorProperty, EnumProperty, PointerProperty, CollectionProperty
)
from bpy.types import PropertyGroup, AddonPreferences

# ------------------------------------------------------------
# Addon Preferences
# ------------------------------------------------------------

class MathPlaygroundPreferences(AddonPreferences):
    """Preferences for Math Playground add-on"""
    bl_idname = __package__
    
    # General Settings
    show_debug_info: BoolProperty(
        name="Show Debug Info",
        description="Show additional debug information in the console",
        default=False
    )
    
    # Performance Settings
    use_instancing: BoolProperty(
        name="Use Instancing",
        description="Use object instancing for better performance with many objects",
        default=True
    )
    
    max_objects: IntProperty(
        name="Maximum Objects",
        description="Maximum number of objects to create before switching to simplified representation",
        default=1000,
        min=100,
        max=10000
    )
    
    # UI Settings
    use_live_preview: BoolProperty(
        name="Live Preview",
        description="Show live preview of objects before creation",
        default=True
    )
    
    default_quality: EnumProperty(
        name="Default Quality",
        description="Default quality setting for created objects",
        items=[
            ('LOW', "Low", "Low quality, fast to create"),
            ('MEDIUM', "Medium", "Medium quality, balanced"),
            ('HIGH', "High", "High quality, slower to create")
        ],
        default='MEDIUM'
    )
    
    def draw(self, context):
        layout = self.layout
        
        # General settings
        box = layout.box()
        box.label(text="General Settings")
        box.prop(self, "show_debug_info")
        
        # Performance settings
        box = layout.box()
        box.label(text="Performance Settings")
        box.prop(self, "use_instancing")
        box.prop(self, "max_objects")
        
        # UI settings
        box = layout.box()
        box.label(text="UI Settings")
        box.prop(self, "use_live_preview")
        box.prop(self, "default_quality")
        
        # About section
        layout.separator()
        layout.label(text="Math Playground Add-on v2.0.0")
        layout.label(text="Explore mathematical concepts in Blender's 3D environment")

# ------------------------------------------------------------
# Module Properties
# ------------------------------------------------------------

class MathPlaygroundLinearAlgebraProperties(PropertyGroup):
    """Properties for Linear Algebra module"""
    
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
    
    show_basis_vectors: BoolProperty(
        name="Show Basis Vectors",
        description="Show the standard basis vectors (i, j, k)",
        default=True
    )
    
    basis_scale: FloatProperty(
        name="Basis Scale",
        description="Scale factor for basis vectors",
        default=1.0,
        min=0.1,
        max=10.0
    )

class MathPlaygroundNumberTheoryProperties(PropertyGroup):
    """Properties for Number Theory module"""
    
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
    
    visualization_mode: EnumProperty(
        name="Visualization Mode",
        description="How to visualize the numbers",
        items=[
            ('3D_BARS', "3D Bars", "Visualize as 3D bars"),
            ('SPIRAL', "Spiral", "Arrange in an Ulam spiral"),
            ('NUMBER_LINE', "Number Line", "Show on a number line"),
            ('SCATTER', "Scatter Plot", "Points in 2D/3D space")
        ],
        default='3D_BARS'
    )

class MathPlaygroundAnalysisProperties(PropertyGroup):
    """Properties for Analysis module"""
    
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
    
    quality: EnumProperty(
        name="Resolution",
        description="Resolution of the plot",
        items=[
            ('LOW', "Low", "Low resolution (faster)"),
            ('MEDIUM', "Medium", "Medium resolution"),
            ('HIGH', "High", "High resolution (slower)")
        ],
        default='MEDIUM'
    )
    
    colormap: EnumProperty(
        name="Color Map",
        description="Color scheme for surface plots",
        items=[
            ('RAINBOW', "Rainbow", "Classic rainbow colors"),
            ('VIRIDIS', "Viridis", "Perceptually uniform blue-green-yellow"),
            ('MAGMA', "Magma", "Perceptually uniform black-red-white"),
            ('GRAYSCALE', "Grayscale", "Grayscale from dark to light")
        ],
        default='VIRIDIS'
    )

class MathPlaygroundGraphTheoryProperties(PropertyGroup):
    """Properties for Graph Theory module"""
    
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
    
    layout_algorithm: EnumProperty(
        name="Layout Algorithm",
        description="Algorithm to position nodes",
        items=[
            ('CIRCLE', "Circle", "Arrange nodes in a circle"),
            ('SPRING', "Force-Directed", "Use force-directed layout"),
            ('GRID', "Grid", "Arrange in a grid pattern"),
            ('SPHERE', "Sphere", "Arrange on the surface of a sphere")
        ],
        default='CIRCLE'
    )
    
    interactive_editing: BoolProperty(
        name="Interactive Editing",
        description="Allow interactive adding and removing of nodes and edges",
        default=False
    )

class MathPlaygroundDifferentialEquationsProperties(PropertyGroup):
    """Properties for Differential Equations module"""
    
    equation: StringProperty(
        name="Equation",
        description="Differential equation in the form dy/dx = f(x,y)",
        default="y",
    )
    
    initial_value: FloatProperty(
        name="Initial Value",
        description="Initial value y(x0)",
        default=1.0,
    )
    
    x0: FloatProperty(
        name="Starting X",
        description="Starting x value",
        default=0.0,
    )
    
    x_max: FloatProperty(
        name="Ending X",
        description="Ending x value",
        default=10.0,
    )
    
    method: EnumProperty(
        name="Solver Method",
        description="Numerical method to solve the equation",
        items=[
            ('EULER', "Euler", "Euler's method (simple but less accurate)"),
            ('RK4', "Runge-Kutta 4", "4th order Runge-Kutta (more accurate)"),
            ('ADAPTIVE', "Adaptive Step", "Adaptive step size (most accurate)")
        ],
        default='RK4'
    )

class MathPlaygroundComplexAnalysisProperties(PropertyGroup):
    """Properties for Complex Analysis module"""
    
    function: StringProperty(
        name="Complex Function",
        description="Complex function f(z) - use z as variable",
        default="z**2",
    )
    
    visualization: EnumProperty(
        name="Visualization Type",
        description="How to visualize the complex function",
        items=[
            ('DOMAIN_COLORING', "Domain Coloring", "Color based on argument and modulus"),
            ('RIEMANN_SPHERE', "Riemann Sphere", "Visualize on the Riemann sphere"),
            ('VECTOR_FIELD', "Vector Field", "Show as a vector field"),
            ('3D_PLOT', "3D Plot", "Plot |f(z)| as height")
        ],
        default='DOMAIN_COLORING'
    )
    
    grid_size: IntProperty(
        name="Grid Size",
        description="Resolution of the visualization",
        default=50,
        min=10,
        max=200
    )

class MathPlaygroundFourierProperties(PropertyGroup):
    """Properties for Fourier Series module"""
    
    function: StringProperty(
        name="Function",
        description="Function to approximate with Fourier series",
        default="x if -pi < x < 0 else 0",
    )
    
    num_terms: IntProperty(
        name="Number of Terms",
        description="Number of terms in the Fourier series",
        default=10,
        min=1,
        max=100
    )
    
    animate: BoolProperty(
        name="Animate",
        description="Create animation showing series convergence",
        default=False
    )

class MathPlaygroundExportProperties(PropertyGroup):
    """Properties for export functionality"""
    
    export_format: EnumProperty(
        name="Export Format",
        description="Format to export data",
        items=[
            ('OBJ', "OBJ", "Wavefront OBJ file"),
            ('CSV', "CSV", "Comma-separated values"),
            ('JSON', "JSON", "JavaScript Object Notation"),
            ('PNG', "PNG Image", "Rendered image")
        ],
        default='OBJ'
    )
    
    export_path: StringProperty(
        name="Export Path",
        description="Path to export files",
        default="//",
        subtype='DIR_PATH'
    )

# ------------------------------------------------------------
# Main Property Group
# ------------------------------------------------------------

class MathPlaygroundProperties(PropertyGroup):
    """Main property group for Math Playground"""
    
    # Module-specific properties
    linear_algebra: PointerProperty(type=MathPlaygroundLinearAlgebraProperties)
    number_theory: PointerProperty(type=MathPlaygroundNumberTheoryProperties)
    analysis: PointerProperty(type=MathPlaygroundAnalysisProperties)
    graph_theory: PointerProperty(type=MathPlaygroundGraphTheoryProperties)
    differential: PointerProperty(type=MathPlaygroundDifferentialEquationsProperties)
    complex: PointerProperty(type=MathPlaygroundComplexAnalysisProperties)
    fourier: PointerProperty(type=MathPlaygroundFourierProperties)
    export: PointerProperty(type=MathPlaygroundExportProperties)
    
    # General properties
    active_module: EnumProperty(
        name="Active Module",
        description="Currently active mathematical module",
        items=[
            ('LINEAR_ALGEBRA', "Linear Algebra", "Vectors, matrices, and transformations"),
            ('NUMBER_THEORY', "Number Theory", "Prime numbers and integer sequences"),
            ('ANALYSIS', "Analysis", "Function plots and calculus"),
            ('GRAPH_THEORY', "Graph Theory", "Graphs and network algorithms"),
            ('DIFFERENTIAL', "Differential Equations", "ODE solvers and visualization"),
            ('COMPLEX', "Complex Analysis", "Complex number visualization"),
            ('FOURIER', "Fourier Series", "Fourier series approximation")
        ],
        default='LINEAR_ALGEBRA'
    )
    
    quality: EnumProperty(
        name="Quality",
        description="Quality setting for visualization",
        items=[
            ('LOW', "Low", "Low quality, faster performance"),
            ('MEDIUM', "Medium", "Medium quality, balanced performance"),
            ('HIGH', "High", "High quality, slower performance")
        ],
        default='MEDIUM'
    )
    
    show_axes: BoolProperty(
        name="Show Axes",
        description="Show coordinate axes in visualizations",
        default=True
    )
    
    use_instancing: BoolProperty(
        name="Use Instancing",
        description="Use object instancing for repeated elements",
        default=True
    )

# ------------------------------------------------------------
# Registration Functions
# ------------------------------------------------------------

# All classes that need to be registered
classes = [
    MathPlaygroundPreferences,
    MathPlaygroundLinearAlgebraProperties,
    MathPlaygroundNumberTheoryProperties,
    MathPlaygroundAnalysisProperties,
    MathPlaygroundGraphTheoryProperties,
    MathPlaygroundDifferentialEquationsProperties,
    MathPlaygroundComplexAnalysisProperties,
    MathPlaygroundFourierProperties,
    MathPlaygroundExportProperties,
    MathPlaygroundProperties
]

def register():
    """Register property classes"""
    for cls in classes:
        bpy.utils.register_class(cls)
    
    # Register main property group
    bpy.types.Scene.math_playground = PointerProperty(type=MathPlaygroundProperties)
    
    print("Math Playground: Properties registered successfully!")

def unregister():
    """Unregister property classes"""
    # Remove properties
    del bpy.types.Scene.math_playground
    
    # Unregister classes
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    print("Math Playground: Properties unregistered successfully!")

if __name__ == "__main__":
    register()