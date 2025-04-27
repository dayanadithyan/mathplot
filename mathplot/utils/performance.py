# utils/performance.py - Performance utilities for Math Playground

import bpy
import numpy as np
import time
import gc
from mathutils import Vector, Matrix
from mathplot.utils import progress

from typing import Callable, List, Optional
# ----------------------------------------
# Object Management Optimization
# ----------------------------------------

# mathplot/utils/performance.py (continued from previous implementation)

# ----------------------------------------
# Batch Processing for Common Operations
# ----------------------------------------


def batch_create_curve(points_list: List[List[Vector]],
                       curve_type: str = 'POLY',
                       thickness: float = 0.01,
                       resolution: int = 4,
                       materials: List = None,
                       collection: Optional[bpy.types.Collection] = None,
                       progress_callback: Optional[Callable[[float,
                                                             str],
                                                            bool]] = None) -> List[bpy.types.Object]:
    """Create multiple curve objects in a batch operation.

    Args:
        points_list: List of point sets, each defining a curve
        curve_type: Type of curve ('POLY', 'BEZIER', 'NURBS')
        thickness: Curve thickness (bevel depth)
        resolution: Bevel resolution
        materials: List of materials to apply to curves (can be None or shorter than points_list)
        collection: Collection to link objects to
        progress_callback: Callback function for progress reporting

    Returns:
        List of created curve objects
    """
    curve_objects = []

    for i, points in enumerate(points_list):
        # Report progress
        if progress_callback:
            progress = (i + 1) / len(points_list)
            if not progress_callback(
                    progress,
                    f"Creating curve {i+1}/{len(points_list)}"):
                break

        # Create curve data
        curve_data = bpy.data.curves.new(f'Curve_{i}', 'CURVE')
        curve_data.dimensions = '3D'

        # Set curve properties
        curve_data.bevel_depth = thickness
        curve_data.bevel_resolution = resolution
        curve_data.use_fill_caps = True

        # Create spline
        spline = curve_data.splines.new(curve_type)

        # Set points
        if curve_type == 'BEZIER':
            # Bezier curve requires special handling
            spline.bezier_points.add(len(points) - 1)
            for j, point in enumerate(points):
                spline.bezier_points[j].co = point
                spline.bezier_points[j].handle_left_type = 'AUTO'
                spline.bezier_points[j].handle_right_type = 'AUTO'
        else:
            # POLY or NURBS curve
            spline.points.add(len(points) - 1)
            for j, point in enumerate(points):
                # Add w=1 for homogeneous coordinates
                spline.points[j].co = (*point, 1)

        # Create curve object
        curve_obj = bpy.data.objects.new(f"Curve_{i}", curve_data)

        # Apply material if available
        if materials and i < len(materials) and materials[i]:
            curve_obj.data.materials.append(materials[i])

        # Link to collection
        if collection:
            collection.objects.link(curve_obj)
        else:
            bpy.context.scene.collection.objects.link(curve_obj)

        curve_objects.append(curve_obj)

    return curve_objects


def batch_create_arrows(origins: List[Vector],
                        directions: List[Vector],
                        length_scale: float = 1.0,
                        shaft_radius: float = 0.05,
                        head_radius: float = 0.1,
                        head_length: float = 0.2,
                        materials: List = None,
                        collection: Optional[bpy.types.Collection] = None,
                        progress_callback: Optional[Callable[[float,
                                                              str],
                                                             bool]] = None) -> List[bpy.types.Object]:
    """Create multiple arrow objects in a batch operation.

    Args:
        origins: List of origin points
        directions: List of direction vectors
        length_scale: Scale factor for arrow length
        shaft_radius: Radius of the arrow shaft
        head_radius: Radius of the arrow head
        head_length: Length of the arrow head as a fraction of total length
        materials: List of materials to apply (can be None or shorter than origins)
        collection: Collection to link objects to
        progress_callback: Callback function for progress reporting

    Returns:
        List of created arrow objects (parent objects containing shaft and head)
    """
    if len(origins) != len(directions):
        raise ValueError(
            "Origins and directions lists must have the same length")

    arrow_objects = []

    for i, (origin, direction) in enumerate(zip(origins, directions)):
        # Report progress
        if progress_callback:
            progress = (i + 1) / len(origins)
            if not progress_callback(progress,
                                     f"Creating arrow {i+1}/{len(origins)}"):
                break

        # Create normalized direction and compute length
        direction_vec = Vector(direction)
        length = direction_vec.length * length_scale

        if length < 0.0001:  # Skip if direction is too small
            continue

        direction_vec.normalize()

        # Create parent empty
        arrow_obj = bpy.data.objects.new(f"Arrow_{i}", None)
        arrow_obj.location = origin

        # Create shaft cylinder
        shaft_length = length * (1 - head_length)
        shaft_mesh = create_cylinder_mesh(
            shaft_radius,
            shaft_length,
            vertices=12,
            cap_ends=True
        )
        shaft_obj = bpy.data.objects.new(f"Arrow_Shaft_{i}", shaft_mesh)

        # Position shaft
        shaft_obj.parent = arrow_obj
        shaft_obj.matrix_parent_inverse = Matrix.Identity(4)
        shaft_obj.location = Vector((0, 0, shaft_length / 2))

        # Create head cone
        head_mesh = create_cone_mesh(
            head_radius,
            length * head_length,
            vertices=12
        )
        head_obj = bpy.data.objects.new(f"Arrow_Head_{i}", head_mesh)

        # Position head
        head_obj.parent = arrow_obj
        head_obj.matrix_parent_inverse = Matrix.Identity(4)
        head_obj.location = Vector(
            (0, 0, shaft_length + (length * head_length / 2)))

        # Orient arrow to direction
        z_axis = Vector((0, 0, 1))
        if direction_vec != z_axis and direction_vec != -z_axis:
            # Calculate rotation to align with direction
            rotation_axis = z_axis.cross(direction_vec)
            rotation_axis.normalize()
            angle = z_axis.angle(direction_vec)
            arrow_obj.rotation_euler = rotation_axis.to_track_quat(
                'Z', 'Y').to_euler()
        elif direction_vec == -z_axis:
            # Handle special case: direction is -Z
            arrow_obj.rotation_euler = (math.pi, 0, 0)

        # Apply material if available
        material = None
        if materials and i < len(materials):
            material = materials[i]

        if material:
            if shaft_obj.data.materials:
                shaft_obj.data.materials[0] = material
            else:
                shaft_obj.data.materials.append(material)

            if head_obj.data.materials:
                head_obj.data.materials[0] = material
            else:
                head_obj.data.materials.append(material)

        # Link to collection
        if collection:
            collection.objects.link(arrow_obj)
            collection.objects.link(shaft_obj)
            collection.objects.link(head_obj)
        else:
            bpy.context.scene.collection.objects.link(arrow_obj)
            bpy.context.scene.collection.objects.link(shaft_obj)
            bpy.context.scene.collection.objects.link(head_obj)

        arrow_objects.append(arrow_obj)

    return arrow_objects


def batch_visualize_points(points: List[Vector],
                           radius: float = 0.05,
                           materials: List = None,
                           collection: Optional[bpy.types.Collection] = None,
                           progress_callback: Optional[Callable[[float, str], bool]] = None,
                           use_instancing: bool = True) -> List[bpy.types.Object]:
    """Create multiple sphere objects to visualize points.

    Args:
        points: List of point positions
        radius: Sphere radius
        materials: List of materials to apply (can be None or shorter than points)
        collection: Collection to link objects to
        progress_callback: Callback function for progress reporting
        use_instancing: Whether to use instancing for better performance

    Returns:
        List of created sphere objects
    """
    if not points:
        return []

    if use_instancing:
        # Create template sphere
        template_material = None
        if materials and len(materials) > 0:
            template_material = materials[0]

        template_data = {
            'type': 'MESH',
            'name': "Point_Template",
            'mesh_data': create_uv_sphere_mesh(radius, 12, 8),
            'material': template_material
        }

        # Create instance data
        instance_data = []
        for i, point in enumerate(points):
            material = None
            if materials and i < len(materials):
                material = materials[i]

            instance_data.append({
                'name': f"Point_{i}",
                'location': point,
                'material': material
            })

        # Create all instances
        template, instances = instancing_create_objects(
            template_data, instance_data, collection, progress_callback)

        return instances
    else:
        # Create individual sphere objects
        object_data = []
        for i, point in enumerate(points):
            material = None
            if materials and i < len(materials):
                material = materials[i]

            object_data.append({
                'type': 'MESH',
                'name': f"Point_{i}",
                'mesh_data': create_uv_sphere_mesh(radius, 12, 8),
                'location': point,
                'material': material
            })

        # Create all spheres
        return batch_create_objects(
            object_data, collection, True, progress_callback)

# ----------------------------------------
# Level of Detail (LOD) Management
# ----------------------------------------


def create_lod_system(base_obj: bpy.types.Object, lod_levels: int = 3,
                      distance_thresholds: List[float] = None) -> List[bpy.types.Object]:
    """Create a complete level of detail system for an object.

    Args:
        base_obj: The high-detail base object
        lod_levels: Number of LOD levels to create
        distance_thresholds: Distance thresholds for each LOD level

    Returns:
        List of LOD objects from highest to lowest detail
    """
    if not base_obj or base_obj.type != 'MESH':
        return []

    # Default distance thresholds if not provided
    if not distance_thresholds:
        distance_thresholds = [10.0 * (i + 1) for i in range(lod_levels - 1)]

    # Ensure we have the right number of thresholds
    if len(distance_thresholds) != lod_levels - 1:
        distance_thresholds = distance_thresholds[:lod_levels - 1]
        while len(distance_thresholds) < lod_levels - 1:
            last = distance_thresholds[-1] if distance_thresholds else 10.0
            distance_thresholds.append(last + 10.0)

    # Create LOD objects
    lod_objects = [base_obj]  # LOD0 is the original object

    for i in range(1, lod_levels):
        # Target reduction increases with LOD level
        target_reduction = 0.3 + (i - 1) * 0.2  # 30%, 50%, 70%, etc.
        target_reduction = min(target_reduction, 0.9)  # Cap at 90% reduction

        # Create LOD mesh
        lod_obj = create_lod_mesh(base_obj, i, target_reduction)
        if lod_obj:
            # Store distance threshold as custom property
            lod_obj["lod_distance"] = distance_thresholds[i - 1]
            lod_objects.append(lod_obj)

    return lod_objects


def setup_lod_drivers(
        lod_objects: List[bpy.types.Object], camera_obj: bpy.types.Object) -> None:
    """Set up drivers to control visibility based on distance from camera.

    Args:
        lod_objects: List of LOD objects from highest to lowest detail
        camera_obj: Camera object to measure distance from
    """
    if not lod_objects or len(lod_objects) < 2 or not camera_obj:
        return

    for i, obj in enumerate(lod_objects):
        # Skip if not a valid object
        if not obj:
            continue

        # Get distance threshold
        if i == 0:  # Base object (LOD0)
            min_dist = 0.0
            max_dist = obj.get("lod_distance", 10.0)
        elif i == len(lod_objects) - 1:  # Lowest detail
            min_dist = lod_objects[i - 1].get("lod_distance", (i - 1) * 10.0)
            max_dist = float('inf')
        else:  # Middle LODs
            min_dist = lod_objects[i - 1].get("lod_distance", (i - 1) * 10.0)
            max_dist = obj.get("lod_distance", i * 10.0)

        # Set up driver for visibility
        driver = obj.driver_add("hide_viewport").driver
        driver.type = 'SCRIPTED'

        # Add variable for distance to camera
        var = driver.variables.new()
        var.name = 'dist'
        var.type = 'LOC_DIFF'
        var.targets[0].id = obj
        var.targets[1].id = camera_obj

        # Set up expression
        if i == 0:  # Base object (highest detail)
            driver.expression = f"dist > {max_dist}"
        elif i == len(lod_objects) - 1:  # Lowest detail
            driver.expression = f"dist < {min_dist}"
        else:  # Middle LODs
            driver.expression = f"dist < {min_dist} or dist > {max_dist}"

        # Also drive render visibility
        render_driver = obj.driver_add("hide_render").driver
        render_driver.type = 'SCRIPTED'
        render_var = render_driver.variables.new()
        render_var.name = 'dist'
        render_var.type = 'LOC_DIFF'
        render_var.targets[0].id = obj
        render_var.targets[1].id = camera_obj
        render_driver.expression = driver.expression
