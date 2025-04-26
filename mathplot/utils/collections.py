# utils/collections.py - Collection utilities for Math Playground

import bpy
from .. import progress

# ----------------------------------------
# Collection Management Functions
# ----------------------------------------
class CollectionHelpers:
    """Helper methods for collection management"""
    @staticmethod
    def safe_get_collection(name):
        return bpy.data.collections.get(name, None)

    @staticmethod
    def purge_empty_collections():
        for col in [c for c in bpy.data.collections if not c.objects]:
            bpy.data.collections.remove(col)
            
def get_collection(name, parent=None):
    """Get or create a collection with the given name.
    
    Args:
        name (str): Name of the collection to get or create
        parent (bpy.types.Collection, optional): Parent collection, defaults to scene collection
        
    Returns:
        bpy.types.Collection: The retrieved or created collection
    """
    # Get existing collection
    collection = bpy.data.collections.get(name)
    
    # Create new collection if it doesn't exist
    if collection is None:
        collection = bpy.data.collections.new(name)
        
        # Link to parent collection
        if parent:
            if isinstance(parent, str):
                parent = get_collection(parent)
            parent.children.link(collection)
        else:
            # Link to scene collection by default
            bpy.context.scene.collection.children.link(collection)
    
    return collection

def clear_collection(collection_name, delete_collection=False):
    """Remove all objects from a collection.
    
    Args:
        collection_name (str): Name of the collection to clear
        delete_collection (bool): Whether to delete the collection after clearing
    """
    # Get the collection
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        return
    
    # Use a list to avoid modifying the collection during iteration
    objects_to_remove = [obj for obj in collection.objects]
    
    # Remove each object
    for i, obj in enumerate(objects_to_remove):
        # Report progress for large collections
        if len(objects_to_remove) > 100:
            progress.report_progress(bpy.context, i / len(objects_to_remove), 
                                    f"Removing object {i+1}/{len(objects_to_remove)}")
        
        # Remove from scene
        bpy.data.objects.remove(obj, do_unlink=True)
    
    # Delete the collection if requested
    if delete_collection and collection:
        bpy.data.collections.remove(collection)
    
    # End progress reporting
    if len(objects_to_remove) > 100:
        progress.end_progress(bpy.context)

def add_object_to_collection(obj, collection_name):
    """Add an object to a collection.
    
    Args:
        obj (bpy.types.Object): Object to add
        collection_name (str): Name of the collection
    """
    # Get or create the collection
    collection = get_collection(collection_name)
    
    # Remove from current collections
    for col in obj.users_collection:
        col.objects.unlink(obj)
    
    # Link to the target collection
    collection.objects.link(obj)

def create_module_collections():
    """Create collections for each math module."""
    # Create main collection
    main = get_collection("Math_Playground")
    
    # Create module collections
    linear_algebra = get_collection("Math_LinearAlgebra", main)
    number_theory = get_collection("Math_NumberTheory", main)
    analysis = get_collection("Math_Analysis", main)
    graph_theory = get_collection("Math_GraphTheory", main)
    differential = get_collection("Math_Differential", main)
    complex_analysis = get_collection("Math_Complex", main)
    fourier = get_collection("Math_Fourier", main)
    
    # Create subcollections for each module
    get_collection("Math_Vectors", linear_algebra)
    get_collection("Math_Matrices", linear_algebra)
    
    get_collection("Math_Primes", number_theory)
    get_collection("Math_Sequences", number_theory)
    
    get_collection("Math_Functions", analysis)
    get_collection("Math_VectorFields", analysis)
    get_collection("Math_Parametric", analysis)
    
    get_collection("Math_Graphs", graph_theory)
    get_collection("Math_GraphAlgorithms", graph_theory)
    
    get_collection("Math_ODE", differential)
    get_collection("Math_PDE", differential)
    
    get_collection("Math_ComplexPlots", complex_analysis)
    get_collection("Math_RiemannSphere", complex_analysis)
    
    get_collection("Math_FourierSeries", fourier)
    get_collection("Math_FourierComponents", fourier)

def clear_module_collections(module_name=None):
    """Clear collections for a specific module or all modules.
    
    Args:
        module_name (str, optional): Module name to clear, or None for all
    """
    # Main collection path
    main_path = "Math_Playground"
    
    # Module collection names and their subcollections
    modules = {
        "LINEAR_ALGEBRA": {
            "path": f"{main_path}/Math_LinearAlgebra",
            "subcollections": ["Math_Vectors", "Math_Matrices"]
        },
        "NUMBER_THEORY": {
            "path": f"{main_path}/Math_NumberTheory",
            "subcollections": ["Math_Primes", "Math_Sequences"]
        },
        "ANALYSIS": {
            "path": f"{main_path}/Math_Analysis",
            "subcollections": ["Math_Functions", "Math_VectorFields", "Math_Parametric"]
        },
        "GRAPH_THEORY": {
            "path": f"{main_path}/Math_GraphTheory",
            "subcollections": ["Math_Graphs", "Math_GraphAlgorithms"]
        },
        "DIFFERENTIAL": {
            "path": f"{main_path}/Math_Differential",
            "subcollections": ["Math_ODE", "Math_PDE"]
        },
        "COMPLEX": {
            "path": f"{main_path}/Math_Complex",
            "subcollections": ["Math_ComplexPlots", "Math_RiemannSphere"]
        },
        "FOURIER": {
            "path": f"{main_path}/Math_Fourier",
            "subcollections": ["Math_FourierSeries", "Math_FourierComponents"]
        }
    }
    
    if module_name:
        # Clear specific module
        if module_name in modules:
            module = modules[module_name]
            # Clear subcollections
            for subcollection in module["subcollections"]:
                clear_collection(f"{module['path']}/{subcollection}")
    else:
        # Clear all module collections
        for module_name, module in modules.items():
            for subcollection in module["subcollections"]:
                clear_collection(f"{module['path']}/{subcollection}")

# ----------------------------------------
# Collection Visibility Functions
# ----------------------------------------

def toggle_collection_visibility(collection_name, visible=True):
    """Toggle the visibility of a collection.
    
    Args:
        collection_name (str): Name of the collection
        visible (bool): Whether to make the collection visible
    """
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Check if the collection is in the view layer
        layer_collection = find_layer_collection(bpy.context.view_layer.layer_collection, collection.name)
        if layer_collection:
            layer_collection.exclude = not visible

def find_layer_collection(layer_collection, name):
    """Recursively find a LayerCollection by name.
    
    Args:
        layer_collection (bpy.types.LayerCollection): The collection to search
        name (str): The name to search for
        
    Returns:
        bpy.types.LayerCollection: The layer collection or None if not found
    """
    if layer_collection.name == name:
        return layer_collection
    
    for child in layer_collection.children:
        found = find_layer_collection(child, name)
        if found:
            return found
    
    return None

def isolate_collection(collection_name):
    """Make only the specified collection visible.
    
    Args:
        collection_name (str): Name of the collection to isolate
    """
    # Get all collections
    layer_collections = get_all_layer_collections(bpy.context.view_layer.layer_collection)
    
    # Hide all collections
    for layer_collection in layer_collections:
        layer_collection.exclude = True
    
    # Show the target collection
    target = find_layer_collection(bpy.context.view_layer.layer_collection, collection_name)
    if target:
        # Also make parent collections visible
        make_parents_visible(target)
        target.exclude = False

def get_all_layer_collections(layer_collection):
    """Get all layer collections recursively.
    
    Args:
        layer_collection (bpy.types.LayerCollection): The root collection
        
    Returns:
        list: List of all layer collections
    """
    collections = [layer_collection]
    
    for child in layer_collection.children:
        collections.extend(get_all_layer_collections(child))
    
    return collections

def make_parents_visible(layer_collection):
    """Make all parent collections visible.
    
    Args:
        layer_collection (bpy.types.LayerCollection): The collection to start from
    """
    # Get the parent collections
    parents = []
    parent = layer_collection.parent
    while parent:
        parents.append(parent)
        parent = parent.parent
    
    # Make all parents visible
    for parent in parents:
        parent.exclude = False

# ----------------------------------------
# Collection Organization Functions
# ----------------------------------------

def organize_objects_by_type(collection_name, create_subcollections=True):
    """Organize objects in a collection by their type.
    
    Args:
        collection_name (str): Name of the collection to organize
        create_subcollections (bool): Whether to create subcollections for each type
    """
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        return
    
    # Group objects by type
    object_groups = {}
    for obj in collection.objects:
        obj_type = obj.type
        if obj_type not in object_groups:
            object_groups[obj_type] = []
        object_groups[obj_type].append(obj)
    
    # Create subcollections and move objects
    for obj_type, objects in object_groups.items():
        if create_subcollections:
            # Create subcollection for this type
            subcollection_name = f"{collection_name}_{obj_type}"
            subcollection = get_collection(subcollection_name, collection)
            
            # Move objects to subcollection
            for obj in objects:
                collection.objects.unlink(obj)
                subcollection.objects.link(obj)
        else:
            # Just group objects by name prefix
            for obj in objects:
                obj.name = f"{obj_type}_{obj.name}"

def create_collection_instance(collection_name, location=(0, 0, 0)):
    """Create an instance of a collection at a specific location.
    
    Args:
        collection_name (str): Name of the collection to instance
        location (tuple): Location for the instance
        
    Returns:
        bpy.types.Object: The created instance object
    """
    collection = bpy.data.collections.get(collection_name)
    if not collection:
        return None
    
    # Create empty object
    empty = bpy.data.objects.new(f"{collection_name}_instance", None)
    empty.location = location
    empty.instance_type = 'COLLECTION'
    empty.instance_collection = collection
    
    # Link to scene
    bpy.context.scene.collection.objects.link(empty)
    
    return empty

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register collection utilities"""
    # Create module collections
    create_module_collections()
    print("Math Playground: Collection utilities registered")

def unregister():
    """Unregister collection utilities"""
    # Clear module collections but don't delete them
    for module_name in ["LINEAR_ALGEBRA", "NUMBER_THEORY", "ANALYSIS", "GRAPH_THEORY", 
                        "DIFFERENTIAL", "COMPLEX", "FOURIER"]:
        clear_module_collections(module_name)
    print("Math Playground: Collection utilities unregistered")