# utils/performance.py - Performance utilities for Math Playground

import bpy
import numpy as np
import time
import gc
from mathutils import Vector, Matrix
from . import progress

# ----------------------------------------
# Object Management Optimization
# ----------------------------------------

def batch_create_objects(object_data, collection=None, link_to_scene=True, progress_callback=None):
    """Create multiple objects in a batch for better performance.
    
    Args:
        object_data (list): List of dictionaries with object creation parameters
                           Each dict should have 'type', 'name', and other type-specific parameters
        collection (bpy.types.Collection, optional): Collection to add objects to
        link_to_scene (bool): Whether to link objects to the scene
        progress_callback (callable, optional): Function to report progress
        
    Returns:
        list: List of created objects
    """
    # Start timer for performance measurement
    start_time = time.time()
    
    # Determine target collection
    if collection is None:
        if link_to_scene:
            collection = bpy.context.scene.collection
    
    # Disable viewport updates for performance
    view_layer = bpy.context.view_layer
    old_hide_viewport = view_layer.hide_viewport
    view_layer.hide_viewport = True
    
    # Prepare objects list
    objects = []
    
    # Process in batches of 100 to avoid memory issues
    batch_size = 100
    total_objects = len(object_data)
    
    for batch_idx in range(0, total_objects, batch_size):
        end_idx = min(batch_idx + batch_size, total_objects)
        batch = object_data[batch_idx:end_idx]
        
        # Create all objects in batch
        for i, data in enumerate(batch):
            # Report progress
            if progress_callback:
                overall_progress = (batch_idx + i) / total_objects
                if not progress_callback(overall_progress, f"Creating object {batch_idx + i + 1}/{total_objects}"):
                    # Restore viewport state
                    view_layer.hide_viewport = old_hide_viewport
                    return objects  # Return what we have so far
            
            # Get object type and name
            obj_type = data.get('type', 'MESH')
            obj_name = data.get('name', f"Object_{batch_idx + i}")
            
            # Create data block based on type
            if obj_type == 'MESH':
                mesh_data = data.get('mesh_data')
                
                # Create a new mesh or use provided one
                if mesh_data is None:
                    mesh = bpy.data.meshes.new(f"{obj_name}_Mesh")
                    
                    # Create mesh from vertices, edges, faces if provided
                    verts = data.get('vertices', [])
                    edges = data.get('edges', [])
                    faces = data.get('faces', [])
                    
                    if verts:
                        mesh.from_pydata(verts, edges, faces)
                        mesh.update()
                else:
                    mesh = mesh_data
                
                # Create object
                obj = bpy.data.objects.new(obj_name, mesh)
                
            elif obj_type == 'CURVE':
                curve_data = data.get('curve_data')
                
                if curve_data is None:
                    curve = bpy.data.curves.new(f"{obj_name}_Curve", 'CURVE')
                    curve.dimensions = data.get('dimensions', '3D')
                    curve.resolution_u = data.get('resolution', 12)
                    curve.bevel_depth = data.get('bevel_depth', 0.0)
                    curve.bevel_resolution = data.get('bevel_resolution', 0)
                    curve.fill_mode = data.get('fill_mode', 'FULL')
                    
                    # Add spline points if provided
                    points = data.get('points', [])
                    if points:
                        spline = curve.splines.new('POLY')
                        spline.points.add(len(points) - 1)
                        for j, point in enumerate(points):
                            spline.points[j].co = (*point, 1)
                else:
                    curve = curve_data
                
                # Create object
                obj = bpy.data.objects.new(obj_name, curve)
                
            elif obj_type == 'EMPTY':
                obj = bpy.data.objects.new(obj_name, None)
                obj.empty_display_type = data.get('empty_display_type', 'PLAIN_AXES')
                obj.empty_display_size = data.get('empty_display_size', 1.0)
                
            else:
                # Unsupported type, skip
                continue
            
            # Set common object properties
            if 'location' in data:
                obj.location = data['location']
            
            if 'rotation' in data:
                if isinstance(data['rotation'], (tuple, list)):
                    obj.rotation_euler = data['rotation']
                elif isinstance(data['rotation'], Matrix):
                    obj.matrix_world = data['rotation']
            
            if 'scale' in data:
                obj.scale = data['scale']
            
            # Set custom properties
            custom_props = data.get('custom_properties', {})
            for prop_name, prop_value in custom_props.items():
                obj[prop_name] = prop_value
            
            # Link object to collection
            if collection:
                collection.objects.link(obj)
            elif link_to_scene:
                bpy.context.scene.collection.objects.link(obj)
            
            # Add object to list
            objects.append(obj)
    
    # Restore viewport state
    view_layer.hide_viewport = old_hide_viewport
    
    # Run garbage collection to free memory
    gc.collect()
    
    # Log performance
    duration = time.time() - start_time
    print(f"Batch created {len(objects)} objects in {duration:.2f} seconds")
    
    return objects

def instancing_create_objects(template_data, instance_data, collection=None, progress_callback=None):
    """Create objects using instancing for better performance.
    
    Args:
        template_data (dict): Dictionary with template object creation parameters
        instance_data (list): List of dictionaries with instance parameters (location, rotation, scale)
        collection (bpy.types.Collection, optional): Collection to add objects to
        progress_callback (callable, optional): Function to report progress
        
    Returns:
        tuple: (template_object, list of instance objects)
    """
    # Start timer for performance measurement
    start_time = time.time()
    
    # Create template object first
    template_objects = batch_create_objects([template_data], collection, True, None)
    if not template_objects:
        return None, []
    
    template_obj = template_objects[0]
    instances = []
    
    # Create instance data for batch creation
    instance_object_data = []
    for i, data in enumerate(instance_data):
        # Basic instance data
        instance = {
            'type': 'EMPTY',
            'name': f"{template_obj.name}_Instance_{i}",
            'empty_display_type': 'PLAIN_AXES',
            'empty_display_size': 0.1,
            'custom_properties': {'instance_source': template_obj.name}
        }
        
        # Add transform data
        if 'location' in data:
            instance['location'] = data['location']
        if 'rotation' in data:
            instance['rotation'] = data['rotation']
        if 'scale' in data:
            instance['scale'] = data['scale']
        
        # Add any additional custom properties
        if 'custom_properties' in data:
            instance['custom_properties'].update(data['custom_properties'])
        
        instance_object_data.append(instance)
    
    # Create all instance objects in batch
    instance_objs = batch_create_objects(instance_object_data, collection, True, progress_callback)
    
    # Set up instancing/duplication
    for inst_obj in instance_objs:
        # Set instancing type
        inst_obj.instance_type = 'OBJECT'
        
        # Set the instance collection
        inst_obj.instance_object = template_obj
        
        instances.append(inst_obj)
    
    # Log performance
    duration = time.time() - start_time
    print(f"Created {len(instances)} instances in {duration:.2f} seconds")
    
    return template_obj, instances

def batch_create_vertices_faces(verts, faces, name="BatchMesh", collection=None):
    """Efficiently create a mesh from a large number of vertices and faces.
    
    Args:
        verts (list): List of vertex coordinates as (x, y, z)
        faces (list): List of face indices
        name (str): Name for the new mesh and object
        collection (bpy.types.Collection, optional): Collection to add the object to
        
    Returns:
        bpy.types.Object: Created mesh object
    """
    # Start timer for performance measurement
    start_time = time.time()
    
    # Create mesh datablock
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    
    # Create mesh from vertices and faces
    mesh.from_pydata(verts, [], faces)
    
    # Calculate normals
    mesh.calc_normals()
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    
    # Link to collection or scene
    if collection:
        collection.objects.link(obj)
    else:
        bpy.context.scene.collection.objects.link(obj)
    
    # Log performance
    num_verts = len(verts)
    num_faces = len(faces)
    duration = time.time() - start_time
    print(f"Created mesh with {num_verts} vertices and {num_faces} faces in {duration:.2f} seconds")
    
    return obj

def optimize_mesh(obj, merge_distance=0.0001, remove_doubles=True):
    """Optimize a mesh by removing doubles, recalculating normals, etc.
    
    Args:
        obj (bpy.types.Object): Object to optimize
        merge_distance (float): Distance for merging vertices
        remove_doubles (bool): Whether to remove duplicate vertices
        
    Returns:
        bool: Success state
    """
    if not obj or obj.type != 'MESH':
        return False
    
    # Store current active object and mode
    old_active = bpy.context.view_layer.objects.active
    old_mode = None
    if old_active:
        old_mode = old_active.mode
    
    # Make target object active
    bpy.context.view_layer.objects.active = obj
    
    # Enter edit mode
    if obj.mode != 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Remove doubles if requested
    if remove_doubles:
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    
    # Recalculate normals
    bpy.ops.mesh.normals_make_consistent(inside=False)
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Update mesh
    obj.data.update()
    
    # Restore previous active object and mode
    if old_active:
        bpy.context.view_layer.objects.active = old_active
        if old_mode and old_mode != 'OBJECT':
            bpy.ops.object.mode_set(mode=old_mode)
    
    return True

# ----------------------------------------
# Batch Processing Optimization
# ----------------------------------------

def batch_process_objects(objects, process_func, batch_size=10, progress_callback=None):
    """Process a batch of objects efficiently.
    
    Args:
        objects (list): List of objects to process
        process_func (callable): Function to process each object
        batch_size (int): Number of objects to process in each batch
        progress_callback (callable, optional): Function to report progress
        
    Returns:
        list: Results from processing each object
    """
    results = []
    total_objects = len(objects)
    
    # Process in batches
    for i in range(0, total_objects, batch_size):
        end_idx = min(i + batch_size, total_objects)
        batch = objects[i:end_idx]
        
        # Process batch
        batch_results = []
        for j, obj in enumerate(batch):
            # Report progress
            if progress_callback:
                overall_progress = (i + j) / total_objects
                if not progress_callback(overall_progress, f"Processing object {i + j + 1}/{total_objects}"):
                    return results  # Return what we have so far
            
            # Process object
            result = process_func(obj)
            batch_results.append(result)
        
        # Extend results
        results.extend(batch_results)
        
        # Run garbage collection between batches
        gc.collect()
    
    return results

# ----------------------------------------
# Memory Management
# ----------------------------------------

def clear_orphaned_data():
    """Clear orphaned data blocks to free memory."""
    # Remove orphaned meshes
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    
    # Remove orphaned materials
    for material in bpy.data.materials:
        if material.users == 0:
            bpy.data.materials.remove(material)
    
    # Remove orphaned textures
    for texture in bpy.data.textures:
        if texture.users == 0:
            bpy.data.textures.remove(texture)
    
    # Remove orphaned images
    for image in bpy.data.images:
        if image.users == 0:
            bpy.data.images.remove(image)
    
    # Force garbage collection
    gc.collect()

def measure_memory_usage():
    """Measure the memory usage of Blender data blocks.
    
    Returns:
        dict: Dictionary with memory usage information
    """
    memory_usage = {
        'meshes': len(bpy.data.meshes),
        'objects': len(bpy.data.objects),
        'materials': len(bpy.data.materials),
        'textures': len(bpy.data.textures),
        'images': len(bpy.data.images),
    }
    
    return memory_usage

# ----------------------------------------
# Level of Detail System
# ----------------------------------------

def create_lod_mesh(obj, lod_level=1, target_reduction=0.5):
    """Create a lower level of detail version of a mesh.
    
    Args:
        obj (bpy.types.Object): Original object
        lod_level (int): Level of detail (higher means lower detail)
        target_reduction (float): Target reduction ratio (0.0-1.0)
        
    Returns:
        bpy.types.Object: New object with reduced detail
    """
    if not obj or obj.type != 'MESH':
        return None
    
    # Create a copy of the original mesh
    lod_mesh = obj.data.copy()
    lod_mesh.name = f"{obj.data.name}_LOD{lod_level}"
    
    # Create new object
    lod_obj = obj.copy()
    lod_obj.data = lod_mesh
    lod_obj.name = f"{obj.name}_LOD{lod_level}"
    
    # Link to same collections
    for collection in obj.users_collection:
        collection.objects.link(lod_obj)
    
    # Store current active object
    old_active = bpy.context.view_layer.objects.active
    
    # Make LOD object active
    bpy.context.view_layer.objects.active = lod_obj
    
    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    # Select all
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Decimate mesh based on LOD level
    detail_ratio = max(0.01, 1.0 - (target_reduction * lod_level))
    bpy.ops.mesh.decimate(ratio=detail_ratio)
    
    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Restore previous active object
    if old_active:
        bpy.context.view_layer.objects.active = old_active
    
    # Hide the LOD object initially
    lod_obj.hide_viewport = True
    lod_obj.hide_render = True
    
    # Store LOD level as custom property
    lod_obj["lod_level"] = lod_level
    lod_obj["original_object"] = obj.name
    
    return lod_obj

def manage_lod_visibility(camera, lod_objects, distance_thresholds=None):
    """Manage the visibility of LOD objects based on distance from camera.
    
    Args:
        camera (bpy.types.Object): Camera object
        lod_objects (dict): Dictionary mapping original objects to lists of LOD objects
        distance_thresholds (list, optional): Distance thresholds for LOD levels
        
    Returns:
        None
    """
    if not camera or camera.type != 'CAMERA':
        return
    
    # Default distance thresholds if not provided
    if distance_thresholds is None:
        distance_thresholds = [10.0, 30.0, 60.0]  # [LOD1, LOD2, LOD3, ...]
    
    # Get camera position
    cam_pos = camera.matrix_world.translation
    
    # Check each original object
    for orig_obj, lod_objs in lod_objects.items():
        if not orig_obj:
            continue
        
        # Calculate distance to camera
        obj_pos = orig_obj.matrix_world.translation
        distance = (obj_pos - cam_pos).length
        
        # Determine which LOD to show based on distance
        show_lod = -1  # -1 means show original
        
        for i, threshold in enumerate(distance_thresholds):
            if distance >= threshold:
                show_lod = i
        
        # Show/hide objects based on LOD level
        orig_obj.hide_viewport = (show_lod >= 0)
        orig_obj.hide_render = (show_lod >= 0)
        
        for i, lod_obj in enumerate(lod_objs):
            if not lod_obj:
                continue
            
            lod_obj.hide_viewport = (show_lod != i)
            lod_obj.hide_render = (show_lod != i)

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register performance utilities"""
    print("Math Playground: Performance utilities registered")

def unregister():
    """Unregister performance utilities"""
    # Clean up orphaned data on unregister
    clear_orphaned_data()
    print("Math Playground: Performance utilities unregistered")