# utils/materials.py - Material utilities for Math Playground

import bpy
import numpy as np
from mathutils import Color

# ----------------------------------------
# Material Creation Functions
# ----------------------------------------

def create_material(name, color, use_nodes=True):
    """Create a new material with the given name and color.
    
    Args:
        name (str): The name for the material
        color (tuple): RGBA color values as a 4-tuple (r, g, b, a)
        use_nodes (bool): Whether to use nodes for the material
        
    Returns:
        bpy.types.Material: The created or retrieved material
    """
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    
    material.use_nodes = use_nodes
    
    if use_nodes:
        # Clear all nodes to start clean
        if 'Principled BSDF' not in material.node_tree.nodes:
            material.node_tree.nodes.clear()
            principled = material.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
            output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
            material.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
        else:
            principled = material.node_tree.nodes['Principled BSDF']
        
        # Ensure color has 4 components (RGBA)
        if len(color) == 3:
            color = (*color, 1.0)  # Add alpha=1.0 if missing
        
        # Set color
        principled.inputs['Base Color'].default_value = color
        
        # Enable transparency if alpha < 1.0
        if color[3] < 1.0:
            material.blend_method = 'BLEND'
            principled.inputs['Alpha'].default_value = color[3]
        else:
            material.blend_method = 'OPAQUE'
    else:
        # Non-node material (faster for simple objects)
        material.diffuse_color = color
    
    return material

def apply_material(obj, material):
    """Apply material to an object.
    
    Args:
        obj (bpy.types.Object): The object to apply material to
        material (bpy.types.Material): The material to apply
    """
    if obj.data and hasattr(obj.data, "materials"):
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)

# ----------------------------------------
# Color Utilities
# ----------------------------------------

def generate_random_color(alpha=1.0, seed=None):
    """Generate a random RGBA color.
    
    Args:
        alpha (float): Alpha value for the color
        seed (int, optional): Random seed for reproducibility
        
    Returns:
        tuple: Random RGBA color as (r, g, b, a)
    """
    if seed is not None:
        np.random.seed(seed)
    
    return (np.random.random(), np.random.random(), np.random.random(), alpha)

def create_colormap(name, num_colors=10, colormap_type='VIRIDIS', alpha=1.0):
    """Create a list of colors from a color map.
    
    Args:
        name (str): Base name for the materials
        num_colors (int): Number of colors to generate
        colormap_type (str): Type of colormap ('VIRIDIS', 'MAGMA', 'RAINBOW', 'GRAYSCALE')
        alpha (float): Alpha value for all colors
        
    Returns:
        list: List of materials with the generated colors
    """
    colors = []
    materials = []
    
    if colormap_type == 'VIRIDIS':
        # Viridis colormap (blue-green-yellow)
        for i in range(num_colors):
            t = i / (num_colors - 1)
            r = 0.267004 + t * 0.278826 + t**2 * 0.134692 + t**3 * 0.047401
            g = 0.004874 + t * 0.757591 + t**2 * 0.263990 + t**3 * 0.046571
            b = 0.329415 + t * 0.096979 + t**2 * 0.165233 + t**3 * 0.035272
            colors.append((r, g, b, alpha))
    
    elif colormap_type == 'MAGMA':
        # Magma colormap (black-red-white)
        for i in range(num_colors):
            t = i / (num_colors - 1)
            r = 0.001462 + t * 2.176424 + t**2 * -1.124781 + t**3 * 0.294596
            g = -0.002299 + t * 0.612417 + t**2 * 0.387921 + t**3 * -0.059461
            b = 0.013866 + t * 1.382501 + t**2 * -0.748825 + t**3 * 0.167693
            colors.append((r, g, b, alpha))
    
    elif colormap_type == 'RAINBOW':
        # Classic rainbow colors (red-orange-yellow-green-blue-indigo-violet)
        for i in range(num_colors):
            hue = i / num_colors
            color = Color()
            color.hsv = (hue, 1.0, 1.0)
            colors.append((color.r, color.g, color.b, alpha))
    
    elif colormap_type == 'GRAYSCALE':
        # Grayscale (black to white)
        for i in range(num_colors):
            val = i / (num_colors - 1)
            colors.append((val, val, val, alpha))
    
    # Create materials
    for i, color in enumerate(colors):
        material = create_material(f"{name}_{i}", color)
        materials.append(material)
    
    return materials

def create_gradient_material(name, start_color, end_color, midpoint_position=0.5):
    """Create a material with a gradient between two colors.
    
    Args:
        name (str): Name for the material
        start_color (tuple): Start color (r,g,b,a)
        end_color (tuple): End color (r,g,b,a)
        midpoint_position (float): Position of the gradient midpoint (0-1)
        
    Returns:
        bpy.types.Material: The created material
    """
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    
    material.use_nodes = True
    node_tree = material.node_tree
    node_tree.nodes.clear()
    
    # Create shader nodes
    output = node_tree.nodes.new('ShaderNodeOutputMaterial')
    principled = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
    gradient = node_tree.nodes.new('ShaderNodeTexGradient')
    color_ramp = node_tree.nodes.new('ShaderNodeValToRGB')
    mapping = node_tree.nodes.new('ShaderNodeMapping')
    tex_coord = node_tree.nodes.new('ShaderNodeTexCoord')
    
    # Set up color ramp
    color_ramp.color_ramp.elements[0].color = start_color
    color_ramp.color_ramp.elements[1].color = end_color
    color_ramp.color_ramp.elements[1].position = midpoint_position
    
    # Connect nodes
    node_tree.links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
    node_tree.links.new(mapping.outputs['Vector'], gradient.inputs['Vector'])
    node_tree.links.new(gradient.outputs['Color'], color_ramp.inputs['Fac'])
    node_tree.links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Enable transparency if needed
    if start_color[3] < 1.0 or end_color[3] < 1.0:
        material.blend_method = 'BLEND'
        node_tree.links.new(color_ramp.outputs['Alpha'], principled.inputs['Alpha'])
    
    return material

def create_wireframe_material(name, color, wire_thickness=0.01, wire_color=None):
    """Create a material that shows wireframe over a base color.
    
    Args:
        name (str): Name for the material
        color (tuple): Base color (r,g,b,a)
        wire_thickness (float): Thickness of the wireframe
        wire_color (tuple, optional): Wireframe color, defaults to black
        
    Returns:
        bpy.types.Material: The created material
    """
    if wire_color is None:
        wire_color = (0.0, 0.0, 0.0, 1.0)
    
    material = bpy.data.materials.get(name)
    if material is None:
        material = bpy.data.materials.new(name)
    
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Clear all nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    wireframe = nodes.new('ShaderNodeWireframe')
    mix_shader = nodes.new('ShaderNodeMixShader')
    wire_principled = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Set wireframe thickness
    wireframe.inputs['Size'].default_value = wire_thickness
    
    # Position nodes
    output.location = (600, 0)
    mix_shader.location = (400, 0)
    wireframe.location = (200, 100)
    principled.location = (200, -100)
    wire_principled.location = (200, 0)
    
    # Set colors
    principled.inputs['Base Color'].default_value = color
    wire_principled.inputs['Base Color'].default_value = wire_color
    
    # Connect nodes
    links.new(wireframe.outputs[0], mix_shader.inputs[0])
    links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
    links.new(wire_principled.outputs['BSDF'], mix_shader.inputs[2])
    links.new(mix_shader.outputs[0], output.inputs['Surface'])
    
    return material

# ----------------------------------------
# Material Registry
# ----------------------------------------

# Dictionary to store commonly used materials
material_registry = {}

def register_common_materials():
    """Create and register commonly used materials"""
    # Clear registry
    material_registry.clear()
    
    # Create axis materials
    material_registry['X_AXIS'] = create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
    material_registry['Y_AXIS'] = create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
    material_registry['Z_AXIS'] = create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
    
    # Create basic vector materials
    material_registry['VECTOR'] = create_material("Vector_Material", (0.8, 0.0, 0.0, 1.0))
    material_registry['VECTOR_I'] = create_material("Vector_I_Material", (1.0, 0.0, 0.0, 1.0))
    material_registry['VECTOR_J'] = create_material("Vector_J_Material", (0.0, 1.0, 0.0, 1.0))
    material_registry['VECTOR_K'] = create_material("Vector_K_Material", (0.0, 0.0, 1.0, 1.0))
    
    # Create number theory materials
    material_registry['PRIME'] = create_material("Prime_Material", (0.0, 0.8, 0.2, 1.0))
    material_registry['SEQUENCE'] = create_material("Sequence_Material", (0.2, 0.4, 0.8, 1.0))
    
    # Create function materials
    material_registry['FUNCTION'] = create_material("Function_Material", (0.0, 0.6, 0.8, 1.0))
    material_registry['PARAMETRIC'] = create_material("Parametric_Material", (0.8, 0.2, 0.8, 1.0))
    material_registry['VECTOR_FIELD'] = create_material("VectorField_Material", (0.8, 0.4, 0.0, 1.0))
    
    # Create graph materials
    material_registry['GRAPH_NODE'] = create_material("Graph_Node_Material", (0.2, 0.6, 0.9, 1.0))
    material_registry['GRAPH_EDGE'] = create_material("Graph_Edge_Material", (0.6, 0.6, 0.6, 1.0))
    material_registry['PATH'] = create_material("Path_Material", (1.0, 0.8, 0.0, 1.0))
    material_registry['MST'] = create_material("MST_Material", (0.0, 1.0, 0.5, 1.0))

def get_material(key):
    """Get a material from the registry by key.
    
    Args:
        key (str): The key for the material
        
    Returns:
        bpy.types.Material: The material, or None if not found
    """
    return material_registry.get(key)

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register material utilities"""
    register_common_materials()
    print("Math Playground: Material utilities registered")

def unregister():
    """Unregister material utilities"""
    material_registry.clear()
    print("Math Playground: Material utilities unregistered")