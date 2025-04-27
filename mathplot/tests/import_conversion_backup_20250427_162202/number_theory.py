# operators/number_theory.py - Number Theory operators

import bpy
import math
import random
import numpy as np
from bpy.types import Operator

from ..utils import materials, progress
from ..utils.collections import get_collection, clear_collection
from ..utils.math_utils import generate_primes, generate_sequence


class MATH_OT_GeneratePrimes(Operator):
    """Generate prime numbers up to a limit"""
    bl_idname = "math.generate_primes"
    bl_label = "Generate Primes"
    bl_options = {'REGISTER', 'UNDO'}

    limit: bpy.props.IntProperty(
        name="Limit",
        description="Generate primes up to this number",
        default=100,
        min=2,
        max=10000,
    )

    arrangement: bpy.props.EnumProperty(
        name="Arrangement",
        description="How to arrange the prime numbers",
        items=[
            ('LINE', "Line", "Arrange primes in a line"),
            ('SPIRAL', "Ulam Spiral", "Arrange primes in an Ulam spiral"),
            ('GRID', "Grid", "Arrange primes in a grid")
        ],
        default='LINE',
    )

    radius: bpy.props.FloatProperty(
        name="Sphere Radius",
        description="Radius of the spheres representing primes",
        default=0.1,
        min=0.01,
        max=1.0,
    )

    spacing: bpy.props.FloatProperty(
        name="Spacing",
        description="Spacing between spheres",
        default=0.5,
        min=0.1,
        max=5.0,
    )

    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    # Create primes collection if it doesn't exist
        collection = get_collection("Math_NumberTheory/Math_Primes")
        
        # Clear existing primes
        clear_collection("Math_NumberTheory/Math_Primes")
        
        # Generate primes
        try:
            progress.start_progress(context, f"Generating primes up to {self.limit}...")
            progress.report_progress(context, 0.0, f"Generating primes up to {self.limit}...")
            
            # Define a progress callback
            def progress_cb(prog, msg):
                    """progress_cb function.
    """
        """progress_cb function.
    """
    progress.report_progress(context, prog, msg)
                return True  # Continue processing
            
            primes = generate_primes(self.limit, progress_cb)
            
            if not primes:
                self.report({'WARNING'}, f"No primes found up to {self.limit}")
                progress.end_progress(context)
                return {'CANCELLED'}
            
            # Create material for primes
            prime_material = materials.create_material("Prime_Material", (0.0, 0.8, 0.2, 1.0))
            
            # Place primes based on arrangement
            if self.arrangement == 'LINE':
                self.create_line_arrangement(context, primes, prime_material, collection)
            elif self.arrangement == 'SPIRAL':
                self.create_spiral_arrangement(context, primes, prime_material, collection)
            elif self.arrangement == 'GRID':
                self.create_grid_arrangement(context, primes, prime_material, collection)
            
            progress.end_progress(context)
            self.report({'INFO'}, f"Generated {len(primes)} prime numbers")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
            self.report({'ERROR'}, f"Error generating primes: {e}")
            return {'CANCELLED'}
    
    def create_line_arrangement(self, context, primes, material, collection):
        """Arrange primes in a line"""
        for i, p in enumerate(primes):
            # Report progress
            progress.report_progress(context, (i + 1) / len(primes), f"Creating prime {i+1}/{len(primes)}")
            
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
            materials.apply_material(sphere, material)
            
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
                if n > 0 and n <= self.limit and n in primes:
                    # Report progress
                    prime_count += 1
                    progress.report_progress(context, prime_count / total_primes, 
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
                    materials.apply_material(sphere, material)
                    
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
            progress.report_progress(context, (i + 1) / len(primes), f"Creating prime {i+1}/{len(primes)}")
            
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
            materials.apply_material(sphere, material)
            
            # Add to collection
            if sphere.users_collection:
                sphere.users_collection[0].objects.unlink(sphere)
            if text.users_collection:
                text.users_collection[0].objects.unlink(text)
            
            collection.objects.link(sphere)
            collection.objects.link(text)
    
    def invoke(self, context, event):
            """invoke function.
    """
        """invoke function.
    """
    # Initialize with current settings
        self.limit = context.scene.math_playground.number_theory.prime_limit
        return self.execute(context)

class MATH_OT_GenerateSequence(Operator):
    """Generate integer sequences"""
    bl_idname = "math.generate_sequence"
    bl_label = "Generate Sequence"
    bl_options = {'REGISTER', 'UNDO'}
    
    sequence_type: bpy.props.EnumProperty(
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
    
    length: bpy.props.IntProperty(
        name="Length",
        description="Number of terms to generate",
        default=10,
        min=1,
        max=100,
    )
    
    custom_formula: bpy.props.StringProperty(
        name="Custom Formula",
        description="Python expression for custom sequence (use n for term index)",
        default="n**2 + 1",
    )
    
    cube_size: bpy.props.FloatProperty(
        name="Cube Size",
        description="Size of cubes representing sequence terms",
        default=0.5,
        min=0.1,
        max=5.0,
    )
    
    spacing: bpy.props.FloatProperty(
        name="Spacing",
        description="Spacing between cubes",
        default=1.0,
        min=0.1,
        max=10.0,
    )
    
    height_scaling: bpy.props.FloatProperty(
        name="Height Scaling",
        description="Scale factor for term value to cube height",
        default=0.1,
        min=0.01,
        max=1.0,
    )
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    # Create sequence collection if it doesn't exist
        collection = get_collection("Math_NumberTheory/Math_Sequences")
        
        # Clear existing sequence
        clear_collection("Math_NumberTheory/Math_Sequences")
        
        try:
            # Generate sequence
            progress.start_progress(context, f"Generating {self.sequence_type} sequence...")
            
            # Define a progress callback
            def progress_cb(prog, msg):
                    """progress_cb function.
    """
        """progress_cb function.
    """
    progress.report_progress(context, prog, msg)
                return True  # Continue processing
            
            sequence = generate_sequence(
                self.sequence_type, 
                self.length, 
                self.custom_formula if self.sequence_type == 'CUSTOM' else None,
                progress_cb
            )
            
            if not sequence:
                self.report({'WARNING'}, "Failed to generate sequence")
                progress.end_progress(context)
                return {'CANCELLED'}
            
            # Create material for sequence
            sequence_material = materials.create_material(
                f"{self.sequence_type}_Sequence_Material", 
                (0.2, 0.4, 0.8, 1.0)
            )
            
            # Place cubes
            for i, term in enumerate(sequence):
                # Report progress
                progress.report_progress(context, (i + 1) / len(sequence), f"Creating term {i+1}/{len(sequence)}")
                
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
                materials.apply_material(cube, sequence_material)
                
                # Add to collection
                if cube.users_collection:
                    cube.users_collection[0].objects.unlink(cube)
                if text.users_collection:
                    text.users_collection[0].objects.unlink(text)
                
                collection.objects.link(cube)
                collection.objects.link(text)
            
            progress.end_progress(context)
            self.report({'INFO'}, f"Generated {len(sequence)} terms of {self.sequence_type} sequence")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
            self.report({'ERROR'}, f"Error generating sequence: {e}")
            return {'CANCELLED'}
    
    def invoke(self, context, event):
            """invoke function.
    """
        """invoke function.
    """
    # Initialize with current settings from scene properties
        props = context.scene.math_playground.number_theory
        self.sequence_type = props.sequence_type
        self.length = props.sequence_length
        self.custom_formula = props.custom_formula
        return self.execute(context)

class MATH_OT_ClearNumberTheory(Operator):
    """Clear all number theory objects from the scene"""
    bl_idname = "math.clear_number_theory"
    bl_label = "Clear Number Theory"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
            """execute function.
    """
        """execute function.
    """
        """execute function.
    """
    clear_collection("Math_NumberTheory/Math_Primes")
        clear_collection("Math_NumberTheory/Math_Sequences")
        self.report({'INFO'}, "All number theory objects cleared")
        return {'FINISHED'}

# Registration functions
classes = [
    MATH_OT_GeneratePrimes,
    MATH_OT_GenerateSequence,
    MATH_OT_ClearNumberTheory,
]

def register():
    """Register Number Theory operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister Number Theory operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
