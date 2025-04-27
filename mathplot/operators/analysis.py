# operators/analysis.py - Analysis operators

import bpy
import math
import numpy as np
from bpy.types import Operator
from mathutils import Vector

from ..utils import materials, progress
from ..utils.collections import get_collection, clear_collection
from ..utils.math_utils import evaluate_expression

class MATH_OT_PlotFunction(Operator):
    """Plot a function y=f(x)"""
    bl_idname = "math.plot_function"
    bl_label = "Plot Function"
    bl_options = {'REGISTER', 'UNDO'}
    
    function: bpy.props.StringProperty(
        name="Function",
        description="Python expression for y=f(x)",
        default="math.sin(x)",
    )
    
    x_min: bpy.props.FloatProperty(
        name="X Min",
        description="Minimum x value",
        default=-10.0,
    )
    
    x_max: bpy.props.FloatProperty(
        name="X Max",
        description="Maximum x value",
        default=10.0,
    )
    
    samples: bpy.props.IntProperty(
        name="Samples",
        description="Number of sample points",
        default=100,
        min=10,
        max=1000,
    )
    
    thickness: bpy.props.FloatProperty(
        name="Curve Thickness",
        description="Thickness of the curve",
        default=0.05,
        min=0.01,
        max=0.5,
    )
    
    color: bpy.props.FloatVectorProperty(
        name="Color",
        description="Curve color",
        default=(0.0, 0.6, 0.8, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype='COLOR',
    )
    
    plot_type: bpy.props.EnumProperty(
        name="Plot Type",
        description="Type of plot to create",
        items=[
            ('CURVE', "Curve", "Plot as a curve"),
            ('SURFACE', "Surface", "Plot as a 3D surface"),
        ],
        default='CURVE',
    )
    
    z_function: bpy.props.StringProperty(
        name="Z Function",
        description="Python expression for z=f(x,y) (for surface plots)",
        default="math.sin(math.sqrt(x**2 + y**2))",
    )
    
    y_min: bpy.props.FloatProperty(
        name="Y Min",
        description="Minimum y value (for surface plots)",
        default=-10.0,
    )
    
    y_max: bpy.props.FloatProperty(
        name="Y Max",
        description="Maximum y value (for surface plots)",
        default=10.0,
    )
    
    def execute(self, context):
        # Create function collection if it doesn't exist
        collection = get_collection("Math_Analysis/Math_Functions")
        
        # Create material for function
        func_material = materials.create_material(f"Function_Material", self.color)
        
        try:
            progress.start_progress(context, "Starting function plot...")
            
            if self.plot_type == 'CURVE':
                self.plot_curve(context, func_material, collection)
            elif self.plot_type == 'SURFACE':
                self.plot_surface(context, func_material, collection)
            
            progress.end_progress(context)
            self.report({'INFO'}, f"Function plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
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
            progress.report_progress(context, (i + 1) / len(x_range) * 0.7, 
                                   f"Evaluating function: point {i+1}/{len(x_range)}")
            
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
        
        progress.report_progress(context, 0.8, "Creating curve object...")
        
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
        materials.apply_material(curve_obj, material)
        
        progress.report_progress(context, 0.9, "Creating coordinate axes...")
        
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
                progress.report_progress(context, point_count / total_points * 0.7, 
                                       f"Evaluating surface: point {point_count}/{total_points}")
                
                try:
                    Z[i, j] = evaluate_expression(self.z_function, {"x": X[i, j], "y": Y[i, j]})
                    # Check for valid result
                    if np.isnan(Z[i, j]) or np.isinf(Z[i, j]):
                        Z[i, j] = 0
                except ValueError:
                    Z[i, j] = 0
        
        progress.report_progress(context, 0.8, "Creating surface mesh...")
        
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
        materials.apply_material(surface_obj, material)
        
        # Set smooth shading
        bpy.context.view_layer.objects.active = surface_obj
        bpy.ops.object.shade_smooth()
        
        progress.report_progress(context, 0.9, "Creating coordinate axes...")
        
        # Add axes
        self.create_axes(is_3d=True)
    
    def create_axes(self, is_3d=False):
        """Create coordinate axes"""
        collection = get_collection("Math_Analysis/Math_Functions")
        
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
        x_material = materials.create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        z_material = materials.create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        materials.apply_material(x_axis, x_material)
        materials.apply_material(z_axis, z_material)
        
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
            y_material = materials.create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
            
            # Apply material
            materials.apply_material(y_axis, y_material)
            
            # Add to collection
            for obj in [y_axis, y_label]:
                if obj.users_collection:
                    obj.users_collection[0].objects.unlink(obj)
                collection.objects.link(obj)
    
    def invoke(self, context, event):
        # Initialize with current settings
        props = context.scene.math_playground.analysis
        self.function = props.function_expression
        self.x_min = props.x_min
        self.x_max = props.x_max
        self.samples = props.samples
        return self.execute(context)

class MATH_OT_PlotParametric(Operator):
    """Plot a parametric curve"""
    bl_idname = "math.plot_parametric"
    bl_label = "Plot Parametric Curve"
    bl_options = {'REGISTER', 'UNDO'}
    
    x_function: bpy.props.StringProperty(
        name="X Function",
        description="Python expression for x=f(t)",
        default="math.cos(t)",
    )
    
    y_function: bpy.props.StringProperty(
        name="Y Function",
        description="Python expression for y=f(t)",
        default="math.sin(t)",
    )
    
    z_function: bpy.props.StringProperty(
        name="Z Function",
        description="Python expression for z=f(t)",
        default="t/10",
    )
    
    t_min: bpy.props.FloatProperty(
        name="T Min",
        description="Minimum t value",
        default=0.0,
    )
    
    t_max: bpy.props.FloatProperty(
        name="T Max",
        description="Maximum t value",
        default=6.28,
    )
    
    samples: bpy.props.IntProperty(
        name="Samples",
        description="Number of sample points",
        default=100,
        min=10,
        max=1000,
    )
    
    thickness: bpy.props.FloatProperty(
        name="Curve Thickness",
        description="Thickness of the curve",
        default=0.05,
        min=0.01,
        max=0.5,
    )
    
    color: bpy.props.FloatVectorProperty(
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
        collection = get_collection("Math_Analysis/Math_Parametric")
        
        # Create material for function
        func_material = materials.create_material("Parametric_Material", self.color)
        
        try:
            progress.start_progress(context, "Starting parametric curve plot...")
            
            # Generate points
            t_range = np.linspace(self.t_min, self.t_max, self.samples)
            points = []
            
            # Evaluate function at each point
            for i, t in enumerate(t_range):
                # Report progress
                progress.report_progress(context, (i + 1) / len(t_range) * 0.7, 
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
            
            progress.report_progress(context, 0.8, "Creating curve object...")
            
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
            materials.apply_material(curve_obj, material)
            
            progress.report_progress(context, 0.9, "Creating coordinate axes...")
            
            # Add axes
            self.create_axes()
            
            progress.end_progress(context)
            self.report({'INFO'}, "Parametric curve plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
            self.report({'ERROR'}, f"Error plotting parametric curve: {e}")
            return {'CANCELLED'}
    
    def create_axes(self):
        """Create coordinate axes"""
        collection = get_collection("Math_Analysis/Math_Parametric")
        
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
        x_material = materials.create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        y_material = materials.create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
        z_material = materials.create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        materials.apply_material(x_axis, x_material)
        materials.apply_material(y_axis, y_material)
        materials.apply_material(z_axis, z_material)
        
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
    
    x_component: bpy.props.StringProperty(
        name="X Component",
        description="Python expression for x component (use x, y, z)",
        default="-y",
    )
    
    y_component: bpy.props.StringProperty(
        name="Y Component",
        description="Python expression for y component (use x, y, z)",
        default="x",
    )
    
    z_component: bpy.props.StringProperty(
        name="Z Component",
        description="Python expression for z component (use x, y, z)",
        default="0",
    )
    
    x_min: bpy.props.FloatProperty(
        name="X Min",
        description="Minimum x value",
        default=-5.0,
    )
    
    x_max: bpy.props.FloatProperty(
        name="X Max",
        description="Maximum x value",
        default=5.0,
    )
    
    y_min: bpy.props.FloatProperty(
        name="Y Min",
        description="Minimum y value",
        default=-5.0,
    )
    
    y_max: bpy.props.FloatProperty(
        name="Y Max",
        description="Maximum y value",
        default=5.0,
    )
    
    z_min: bpy.props.FloatProperty(
        name="Z Min",
        description="Minimum z value",
        default=-5.0,
    )
    
    z_max: bpy.props.FloatProperty(
        name="Z Max",
        description="Maximum z value",
        default=5.0,
    )
    
    grid_size: bpy.props.IntProperty(
        name="Grid Size",
        description="Number of grid points in each dimension",
        default=5,
        min=2,
        max=20,
    )
    
    arrow_scale: bpy.props.FloatProperty(
        name="Arrow Scale",
        description="Scale factor for arrows",
        default=0.5,
        min=0.1,
        max=5.0,
    )
    
    normalize: bpy.props.BoolProperty(
        name="Normalize Vectors",
        description="Normalize vector lengths",
        default=False,
    )
    
    color: bpy.props.FloatVectorProperty(
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
        collection = get_collection("Math_Analysis/Math_VectorFields")
        
        # Clear existing vector field
        clear_collection("Math_Analysis/Math_VectorFields")
        
        # Create material for vector field
        field_material = materials.create_material("VectorField_Material", self.color)
        
        try:
            progress.start_progress(context, "Starting vector field plot...")
            
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
                        progress.report_progress(context, point_count / total_points * 0.8, 
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
                            continue
            
            progress.report_progress(context, 0.9, "Creating coordinate axes...")
            
            # Add axes
            self.create_axes()
            
            progress.end_progress(context)
            self.report({'INFO'}, "Vector field plotted successfully")
            return {'FINISHED'}
            
        except Exception as e:
            progress.end_progress(context)
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
        materials.apply_material(shaft, material)
        materials.apply_material(head, material)
        
        # Add to collection
        if shaft.users_collection:
            shaft.users_collection[0].objects.unlink(shaft)
        if head.users_collection:
            head.users_collection[0].objects.unlink(head)
        
        collection.objects.link(shaft)
        collection.objects.link(head)
    
    def create_axes(self):
        """Create coordinate axes"""
        collection = get_collection("Math_Analysis/Math_VectorFields")
        
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
        x_material = materials.create_material("X_Axis_Material", (1.0, 0.2, 0.2, 1.0))
        y_material = materials.create_material("Y_Axis_Material", (0.2, 1.0, 0.2, 1.0))
        z_material = materials.create_material("Z_Axis_Material", (0.2, 0.2, 1.0, 1.0))
        
        # Apply materials
        materials.apply_material(x_axis, x_material)
        materials.apply_material(y_axis, y_material)
        materials.apply_material(z_axis, z_material)
        
        # Add to collection
        for obj in [x_axis, x_label, y_axis, y_label, z_axis, z_label]:
            if obj.users_collection:
                obj.users_collection[0].objects.unlink(obj)
            collection.objects.link(obj)
    
    def invoke(self, context, event):
        # Initialize with current settings
        props = context.scene.math_playground.analysis
        self.x_component = props.vector_field_x
        self.y_component = props.vector_field_y
        self.z_component = props.vector_field_z
        return self.execute(context)

class MATH_OT_ClearAnalysis(Operator):
    """Clear all analysis objects from the scene"""
    bl_idname = "math.clear_analysis"
    bl_label = "Clear Analysis"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        clear_collection("Math_Analysis/Math_Functions")
        clear_collection("Math_Analysis/Math_VectorFields")
        clear_collection("Math_Analysis/Math_Parametric")
        self.report({'INFO'}, "All analysis objects cleared")
        return {'FINISHED'}

# Registration functions
classes = [
    MATH_OT_PlotFunction,
    MATH_OT_PlotParametric,
    MATH_OT_PlotVectorField,
    MATH_OT_ClearAnalysis,
]

def register():
    """Register Analysis operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister Analysis operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)