import bpy
from bpy.props import (
    StringProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    EnumProperty,
    FloatVectorProperty,
    PointerProperty
)
from bpy.types import PropertyGroup

# Linear Algebra Properties
class LinearAlgebraPropertyGroup(PropertyGroup):
    """Property group for Linear Algebra module"""
    
    vector_color: FloatVectorProperty(
        name="Vector Color",
        description="Color of the vector",
        default=(1.0, 0.0, 0.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR'
    )
    
    matrix_input: StringProperty(
        name="Matrix",
        description="3x3 matrix in format 'a,b,c;d,e,f;g,h,i'",
        default="1,0,0;0,1,0;0,0,1"
    )
    
    vector_scale: FloatProperty(
        name="Vector Scale",
        description="Scale factor for vector visualization",
        default=1.0,
        min=0.1,
        max=10.0
    )

# Number Theory Properties
class NumberTheoryPropertyGroup(PropertyGroup):
    """Property group for Number Theory module"""
    
    prime_limit: IntProperty(
        name="Prime Limit",
        description="Generate primes up to this number",
        default=100,
        min=2,
        max=10000
    )
    
    sequence_length: IntProperty(
        name="Sequence Length",
        description="Number of terms to generate",
        default=10,
        min=1,
        max=100
    )
    
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
        default='FIBONACCI'
    )
    
    custom_formula: StringProperty(
        name="Custom Formula",
        description="Python expression for custom sequence (use n for term index)",
        default="n**2 + 1"
    )

# Analysis Properties
class AnalysisPropertyGroup(PropertyGroup):
    """Property group for Analysis module"""
    
    function_expression: StringProperty(
        name="Function",
        description="Python expression for y=f(x)",
        default="math.sin(x)"
    )
    
    x_min: FloatProperty(
        name="X Min",
        description="Minimum x value",
        default=-10.0
    )
    
    x_max: FloatProperty(
        name="X Max",
        description="Maximum x value",
        default=10.0
    )
    
    samples: IntProperty(
        name="Samples",
        description="Number of sample points",
        default=100,
        min=10,
        max=1000
    )
    
    vector_field_x: StringProperty(
        name="X Component",
        description="Python expression for x component (use x, y, z)",
        default="-y"
    )
    
    vector_field_y: StringProperty(
        name="Y Component",
        description="Python expression for y component (use x, y, z)",
        default="x"
    )
    
    vector_field_z: StringProperty(
        name="Z Component",
        description="Python expression for z component (use x, y, z)",
        default="0"
    )

# Graph Theory Properties
class GraphTheoryPropertyGroup(PropertyGroup):
    """Property group for Graph Theory module"""
    
    node_count: IntProperty(
        name="Node Count",
        description="Number of nodes in the graph",
        default=10,
        min=2,
        max=100
    )
    
    edge_probability: FloatProperty(
        name="Edge Probability",
        description="Probability of creating an edge between two nodes",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )
    
    layout_type: EnumProperty(
        name="Layout Type",
        description="Algorithm for arranging graph nodes",
        items=[
            ('CIRCLE', "Circle", "Arrange nodes in a circle"),
            ('RANDOM', "Random", "Arrange nodes randomly"),
            ('FORCE_DIRECTED', "Force Directed", "Use force-directed layout algorithm")
        ],
        default='CIRCLE'
    )

# Main Property Group
class MathPlaygroundPropertyGroup(PropertyGroup):
    """Main property group for Math Playground"""
    
    active_module: EnumProperty(
        name="Active Module",
        description="Currently active math module",
        items=[
            ('LINEAR_ALGEBRA', "Linear Algebra", "Linear Algebra tools"),
            ('NUMBER_THEORY', "Number Theory", "Number Theory tools"),
            ('ANALYSIS', "Analysis", "Analysis tools"),
            ('GRAPH_THEORY', "Graph Theory", "Graph Theory tools")
        ],
        default='LINEAR_ALGEBRA'
    )
    
    # Sub-property groups
    linear_algebra: PointerProperty(type=LinearAlgebraPropertyGroup)
    number_theory: PointerProperty(type=NumberTheoryPropertyGroup)
    analysis: PointerProperty(type=AnalysisPropertyGroup)
    graph_theory: PointerProperty(type=GraphTheoryPropertyGroup)

# All classes that need to be registered
classes = [
    LinearAlgebraPropertyGroup,
    NumberTheoryPropertyGroup,
    AnalysisPropertyGroup,
    GraphTheoryPropertyGroup,
    MathPlaygroundPropertyGroup
]

def register():
    """Register all property classes"""
    for cls in classes:
        bpy.utils.register_class(cls)
    print("Math Playground: Properties registered")

def unregister():
    """Unregister all property classes"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Math Playground: Properties unregistered")