# operators/graph_theory.py - Graph Theory operators

import bpy
import math
import random
import numpy as np
from bpy.types import Operator
from mathutils import Vector
from ..utils import materials, progress
from ..utils.collections import get_collection, clear_collection


class MATH_OT_CreateGraph(Operator):
    """Create a graph with nodes and edges"""
    bl_idname = "math.create_graph"
    bl_label = "Create Graph"
    bl_options = {'REGISTER', 'UNDO'}

    node_count: bpy.props.IntProperty(
        name="Node Count",
        description="Number of nodes in the graph",
        default=10,
        min=2,
        max=100
    )

    edge_probability: bpy.props.FloatProperty(
        name="Edge Probability",
        description="Probability of creating an edge between two nodes",
        default=0.3,
        min=0.0,
        max=1.0,
        subtype='FACTOR'
    )

    layout_type: bpy.props.EnumProperty(
        name="Layout Type",
        description="Algorithm for arranging graph nodes",
        items=[
            ('CIRCLE', "Circle", "Arrange nodes in a circle"),
            ('RANDOM', "Random", "Arrange nodes randomly"),
            ('FORCE_DIRECTED', "Force Directed",
             "Use force-directed layout algorithm")
        ],
        default='CIRCLE'
    )

    node_size: bpy.props.FloatProperty(
        name="Node Size",
        description="Size of the graph nodes",
        default=0.2,
        min=0.05,
        max=1.0
    )

    edge_thickness: bpy.props.FloatProperty(
        name="Edge Thickness",
        description="Thickness of the graph edges",
        default=0.05,
        min=0.01,
        max=0.2
    )

    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    # Create graph collection if it doesn't exist
        collection = get_collection("Math_GraphTheory/Math_Graphs")
        
        # Clear existing graph
        clear_collection("Math_GraphTheory/Math_Graphs")
        
        try:
            # Start progress reporting
            progress.start_progress(context, "Creating graph...")
            
            # Create materials
            node_material = materials.create_material("Graph_Node_Material", (0.2, 0.6, 0.9, 1.0))
            edge_material = materials.create_material("Graph_Edge_Material", (0.6, 0.6, 0.6, 1.0))
            
            # Generate node positions based on layout
            node_positions = self.generate_layout(context)
            
            # Create nodes
            nodes = []
            for i, pos in enumerate(node_positions):
                # Report progress
                progress.report_progress(context, i / self.node_count * 0.5, f"Creating node {i+1}/{self.node_count}")
                
                # Create sphere for node
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=self.node_size,
                    location=pos,
                    segments=16,
                    ring_count=8
                )
                node_obj = bpy.context.active_object
                node_obj.name = f"Node_{i}"
                
                # Apply material
                materials.apply_material(node_obj, node_material)
                
                # Add to collection
                if node_obj.users_collection:
                    node_obj.users_collection[0].objects.unlink(node_obj)
                collection.objects.link(node_obj)
                
                nodes.append(node_obj)
            
            # Generate random edges
            edges_created = 0
            potential_edges = [(i, j) for i in range(self.node_count) for j in range(i+1, self.node_count)]
            
            for edge_idx, (i, j) in enumerate(potential_edges):
                # Report progress
                progress.report_progress(context, 0.5 + edge_idx / len(potential_edges) * 0.5, 
                                     f"Processing edge {edge_idx+1}/{len(potential_edges)}")
                
                # Randomly determine if this edge should be created
                if random.random() < self.edge_probability:
                    self.create_edge(nodes[i], nodes[j], edge_material, collection)
                    edges_created += 1
            
            progress.end_progress(context)
            self.report({'INFO'}, f"Created graph with {self.node_count} nodes and {edges_created} edges")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
            self.report({'ERROR'}, f"Error creating graph: {e}")
            return {'CANCELLED'}
    
    def generate_layout(self, context):
        """Generate node positions based on selected layout algorithm"""
        positions = []
        
        # Progress callback for iterative layouts
        def progress_cb(prog, msg):
                """progress_cb function.
    """
    progress.report_progress(context, prog * 0.2, msg)
            return True  # Continue processing
        
        if self.layout_type == 'CIRCLE':
            # Arrange nodes in a circle
            radius = self.node_count * 0.2
            for i in range(self.node_count):
                angle = 2 * math.pi * i / self.node_count
                pos = (radius * math.cos(angle), radius * math.sin(angle), 0)
                positions.append(pos)
                
        elif self.layout_type == 'RANDOM':
            # Random layout
            area_size = self.node_count * 0.3
            for i in range(self.node_count):
                pos = (
                    random.uniform(-area_size, area_size),
                    random.uniform(-area_size, area_size),
                    random.uniform(-area_size/4, area_size/4)
                )
                positions.append(pos)
                
        elif self.layout_type == 'FORCE_DIRECTED':
            # Simple force-directed layout (simplified)
            # Start with random positions
            area_size = self.node_count * 0.3
            positions = [
                (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.5, 0.5))
                for _ in range(self.node_count)
            ]
            
            # Run iterations to improve layout
            iterations = 50
            for iter in range(iterations):
                # Report progress
                if not progress_cb(iter / iterations, f"Optimizing layout: iteration {iter+1}/{iterations}"):
                    break
                
                # Calculate repulsive forces between all nodes
                forces = [(0, 0, 0) for _ in range(self.node_count)]
                
                # Repulsive forces between nodes
                for i in range(self.node_count):
                    for j in range(self.node_count):
                        if i != j:
                            # Calculate vector from j to i
                            dx = positions[i][0] - positions[j][0]
                            dy = positions[i][1] - positions[j][1]
                            dz = positions[i][2] - positions[j][2]
                            
                            # Distance (prevent division by zero)
                            dist = max(0.1, math.sqrt(dx*dx + dy*dy + dz*dz))
                            
                            # Repulsive force inversely proportional to distance
                            force = 1.0 / (dist * dist)
                            
                            # Update forces
                            fx, fy, fz = forces[i]
                            forces[i] = (
                                fx + dx/dist * force,
                                fy + dy/dist * force,
                                fz + dz/dist * force
                            )
                
                # Apply forces to update positions
                damping = 0.1 * (1 - iter/iterations)  # Reduce movement over time
                for i in range(self.node_count):
                    fx, fy, fz = forces[i]
                    positions[i] = (
                        positions[i][0] + fx * damping,
                        positions[i][1] + fy * damping,
                        positions[i][2] + fz * damping
                    )
            
            # Scale positions
            positions = [(x*area_size, y*area_size, z*area_size) for x, y, z in positions]
        
        return positions
    
    def create_edge(self, node1, node2, material, collection):
        """Create an edge between two nodes"""
        # Get node centers
        p1 = node1.location
        p2 = node2.location
        
        # Calculate edge center and length
        center = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2, (p1[2] + p2[2])/2)
        length = math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)
        
        # Create cylinder for edge
        bpy.ops.mesh.primitive_cylinder_add(
            radius=self.edge_thickness,
            depth=length,
            location=center
        )
        edge_obj = bpy.context.active_object
        edge_obj.name = f"Edge_{node1.name}_{node2.name}"
        
        # Orient cylinder to point from node1 to node2
        direction = (p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2])
        
        # Convert direction to rotation
        z_axis = Vector((0, 0, 1))
        direction_vec = Vector(direction).normalized()
        
        if direction_vec.length > 0:
            rot_axis = z_axis.cross(direction_vec)
            if rot_axis.length > 0:
                rot_angle = math.acos(min(1, max(-1, z_axis.dot(direction_vec))))
                rot_axis.normalize()
                edge_obj.rotation_euler = rot_axis.to_track_quat('Z', 'Y').to_euler()
        
        # Apply material
        materials.apply_material(edge_obj, material)
        
        # Add to collection
        if edge_obj.users_collection:
            edge_obj.users_collection[0].objects.unlink(edge_obj)
        collection.objects.link(edge_obj)
        
        return edge_obj
    
    def invoke(self, context, event):
            """invoke function.
    """
    # Initialize with current settings from scene properties
        props = context.scene.math_playground.graph_theory
        self.node_count = props.node_count
        self.edge_probability = props.edge_probability
        self.layout_type = props.layout_type
        return self.execute(context)

class MATH_OT_FindShortestPath(Operator):
    """Find the shortest path between two nodes in a graph"""
    bl_idname = "math.find_shortest_path"
    bl_label = "Find Shortest Path"
    bl_options = {'REGISTER', 'UNDO'}
    
    start_node: bpy.props.IntProperty(
        name="Start Node",
        description="Index of the start node",
        default=0,
        min=0
    )
    
    end_node: bpy.props.IntProperty(
        name="End Node",
        description="Index of the end node",
        default=1,
        min=0
    )
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    # This is a placeholder implementation
        # A full implementation would:
        # 1. Extract the graph structure from the scene
        # 2. Run Dijkstra's algorithm
        # 3. Highlight the path
        
        self.report({'INFO'}, f"Shortest path feature not yet implemented")
        return {'FINISHED'}

class MATH_OT_ClearGraphTheory(Operator):
    """Clear all graph theory objects from the scene"""
    bl_idname = "math.clear_graph_theory"
    bl_label = "Clear Graph Theory"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    clear_collection("Math_GraphTheory/Math_Graphs")
        clear_collection("Math_GraphTheory/Math_GraphAlgorithms")
        self.report({'INFO'}, "All graph theory objects cleared")
        return {'FINISHED'}

# Registration functions
classes = [
    MATH_OT_CreateGraph,
    MATH_OT_FindShortestPath,
    MATH_OT_ClearGraphTheory,
]

def register():
    """Register Graph Theory operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister Graph Theory operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
