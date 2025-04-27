# utils/error_utils.py - Error handling utilities for Math Playground

import bpy
import traceback
import inspect
import numpy as np
from mathutils import Vector, Matrix
import logging

# Set up logging
logger = logging.getLogger("MathPlayground")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ----------------------------------------
# Error Checking Functions
# ----------------------------------------

def validate_context(context, required_attrs=None):
    """Validate that context is valid and has required attributes.
    
    Args:
        context (bpy.types.Context): Context to validate
        required_attrs (list, optional): List of required attributes
        
    Returns:
        bool: True if valid, False otherwise
        
    Raises:
        ValueError: If context is invalid and raise_exception is True
    """
    if context is None:
        logger.error("Context is None")
        return False
    
    if required_attrs:
        for attr in required_attrs:
            if not hasattr(context, attr):
                logger.error(f"Context missing required attribute: {attr}")
                return False
            
            # Additional checks for commonly used attributes
            if attr == 'area' and context.area is None:
                logger.error("Context has area attribute but it is None")
                return False
                
            if attr == 'scene' and context.scene is None:
                logger.error("Context has scene attribute but it is None")
                return False
    
    return True

def validate_matrix(matrix, size=(4, 4), det_nonzero=True):
    """Validate that a matrix has the correct size and properties.
    
    Args:
        matrix: Matrix to validate (mathutils.Matrix or numpy.ndarray)
        size (tuple): Expected size of the matrix
        det_nonzero (bool): Whether to check if determinant is non-zero
        
    Returns:
        tuple: (valid, error_message)
    """
    # Check type
    if isinstance(matrix, Matrix):
        # Convert to numpy for consistent handling
        matrix_np = np.array(matrix)
    elif isinstance(matrix, np.ndarray):
        matrix_np = matrix
    else:
        return False, f"Matrix must be a mathutils.Matrix or numpy.ndarray, got {type(matrix)}"
    
    # Check size
    if matrix_np.shape != size:
        return False, f"Matrix has incorrect shape {matrix_np.shape}, expected {size}"
    
    # Check if determinant is non-zero (for invertible matrices)
    if det_nonzero and size[0] == size[1]:
        try:
            det = np.linalg.det(matrix_np)
            if abs(det) < 1e-6:
                return False, f"Matrix is singular (determinant ≈ 0)"
        except np.linalg.LinAlgError:
            return False, "Could not compute matrix determinant"
    
    return True, ""

def validate_vector(vector, size=3):
    """Validate that a vector has the correct size.
    
    Args:
        vector: Vector to validate (mathutils.Vector, numpy.ndarray, list, or tuple)
        size (int): Expected size of the vector
        
    Returns:
        tuple: (valid, error_message)
    """
    # Check type and convert to numpy for consistent handling
    if isinstance(vector, Vector):
        vec_np = np.array(vector)
    elif isinstance(vector, np.ndarray):
        vec_np = vector
    elif isinstance(vector, (list, tuple)):
        try:
            vec_np = np.array(vector)
        except ValueError:
            return False, f"Could not convert {type(vector)} to numpy array"
    else:
        return False, f"Vector must be a mathutils.Vector, numpy.ndarray, list, or tuple, got {type(vector)}"
    
    # Check size
    if vec_np.size != size:
        return False, f"Vector has incorrect size {vec_np.size}, expected {size}"
    
    return True, ""

def validate_color(color):
    """Validate that a color has the correct format.
    
    Args:
        color: Color to validate (tuple or list with 3 or 4 components)
        
    Returns:
        tuple: (valid, error_message)
    """
    if not isinstance(color, (tuple, list)):
        return False, f"Color must be a tuple or list, got {type(color)}"
    
    if len(color) < 3 or len(color) > 4:
        return False, f"Color must have 3 or 4 components, got {len(color)}"
    
    for i, component in enumerate(color):
        if not isinstance(component, (int, float)):
            return False, f"Color component {i} must be a number, got {type(component)}"
        if component < 0 or component > 1:
            return False, f"Color component {i} must be between 0 and 1, got {component}"
    
    return True, ""

def validate_expression(expression, allowed_vars=None, test_values=None):
    """Validate that a mathematical expression is safe and evaluates correctly.
    
    Args:
        expression (str): Expression to validate
        allowed_vars (list, optional): List of allowed variable names
        test_values (dict, optional): Test values for variables
        
    Returns:
        tuple: (valid, error_message)
    """
    if not isinstance(expression, str):
        return False, f"Expression must be a string, got {type(expression)}"
    
    # Check for potentially unsafe operations
    unsafe_terms = ["__", "import ", "eval(", "exec(", "compile(", 
                  "globals(", "locals(", "getattr(", "setattr(", 
                  "delattr(", "open(", "file(", "os.", "sys."]
    
    for term in unsafe_terms:
        if term in expression:
            return False, f"Expression contains unsafe term: {term}"
    
    # Check if the expression uses only allowed variables
    if allowed_vars:
        # Extract variable names from the expression (simple approach)
        import re
        var_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        vars_in_expr = set(re.findall(var_pattern, expression))
        
        # Remove Python built-ins and math functions
        import math
        builtin_funcs = dir(math) + ['abs', 'min', 'max', 'pow']
        vars_in_expr = vars_in_expr - set(builtin_funcs)
        
        # Check if any disallowed variables are used
        disallowed_vars = vars_in_expr - set(allowed_vars)
        if disallowed_vars:
            return False, f"Expression uses disallowed variables: {', '.join(disallowed_vars)}"
    
    # Try evaluating the expression with test values if provided
    if test_values:
        try:
            # Provide a safe namespace for evaluation
            safe_namespace = {
                "math": math,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "sqrt": math.sqrt,
                "exp": math.exp,
                "log": math.log,
                "log10": math.log10,
                "pi": math.pi,
                "e": math.e,
                "abs": abs,
                "pow": pow,
                "min": min,
                "max": max,
            }
            
            # Add test values
            safe_namespace.update(test_values)
            
            # Evaluate expression
            result = eval(expression, {"__builtins__": {}}, safe_namespace)
            
            # Check if result is a number
            if not isinstance(result, (int, float, complex, np.ndarray, Vector, Matrix)):
                return False, f"Expression evaluates to {type(result)}, expected a numeric type"
                
            # Check for NaN or Inf
            if isinstance(result, (float, complex)) and (np.isnan(result) or np.isinf(result)):
                return False, f"Expression evaluates to {result}"
                
        except Exception as e:
            return False, f"Error evaluating expression: {str(e)}"
    
    return True, ""

def validate_mesh(mesh):
    """Validate that a mesh is valid.
    
    Args:
        mesh: Mesh to validate (bpy.types.Mesh)
        
    Returns:
        tuple: (valid, error_message)
    """
    if not mesh:
        return False, "Mesh is None"
        
    if not isinstance(mesh, bpy.types.Mesh):
        return False, f"Expected bpy.types.Mesh, got {type(mesh)}"
    
    # Check if mesh has vertices
    if len(mesh.vertices) == 0:
        return False, "Mesh has no vertices"
    
    # Check for degenerate faces
    for face in mesh.polygons:
        if len(face.vertices) < 3:
            return False, f"Mesh contains degenerate face with {len(face.vertices)} vertices"
    
    # Check for non-manifold edges (edges connected to more than 2 faces)
    # This is more complex and would require using bpy.ops.mesh.select_non_manifold
    
    return True, ""

def validate_object(obj):
    """Validate that an object is valid.
    
    Args:
        obj: Object to validate (bpy.types.Object)
        
    Returns:
        tuple: (valid, error_message)
    """
    if not obj:
        return False, "Object is None"
        
    if not isinstance(obj, bpy.types.Object):
        return False, f"Expected bpy.types.Object, got {type(obj)}"
    
    # Check if object exists in the scene
    if not obj.name in bpy.data.objects:
        return False, f"Object named '{obj.name}' does not exist in bpy.data.objects"
    
    # Check if object has valid data
    if obj.type == 'MESH':
        if not obj.data or not isinstance(obj.data, bpy.types.Mesh):
            return False, "Mesh object has invalid or missing mesh data"
            
    elif obj.type == 'CAMERA':
        if not obj.data or not isinstance(obj.data, bpy.types.Camera):
            return False, "Camera object has invalid or missing camera data"
            
    elif obj.type == 'LIGHT':
        if not obj.data or not isinstance(obj.data, bpy.types.Light):
            return False, "Light object has invalid or missing light data"
    
    return True, ""

def validate_range(value, min_val=None, max_val=None, allow_none=False):
    """Validate that a value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        allow_none (bool): Whether None is allowed
        
    Returns:
        tuple: (valid, error_message)
    """
    if value is None:
        if allow_none:
            return True, ""
        else:
            return False, "Value is None"
    
    if not isinstance(value, (int, float)):
        return False, f"Value must be a number, got {type(value)}"
    
    if min_val is not None and value < min_val:
        return False, f"Value {value} is less than the minimum {min_val}"
    
    if max_val is not None and value > max_val:
        return False, f"Value {value} is greater than the maximum {max_val}"
    
    return True, ""

def validate_collection(collection):
    """Validate that a collection is valid.
    
    Args:
        collection: Collection to validate (bpy.types.Collection)
        
    Returns:
        tuple: (valid, error_message)
    """
    if not collection:
        return False, "Collection is None"
        
    if not isinstance(collection, bpy.types.Collection):
        return False, f"Expected bpy.types.Collection, got {type(collection)}"
    
    # Check if collection exists
    if not collection.name in bpy.data.collections:
        return False, f"Collection named '{collection.name}' does not exist in bpy.data.collections"
    
    return True, ""

# ----------------------------------------
# Exception Handling Decorators
# ----------------------------------------

def safely_execute(func):
    """Decorator for safely executing a function with proper exception handling.
    
    Args:
        func (callable): Function to decorate
        
    Returns:
        callable: Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function info
            func_info = f"{func.__module__}.{func.__name__}"
            
            # Get stack trace
            stack_trace = traceback.format_exc()
            
            # Log the error
            logger.error(f"Error in {func_info}: {str(e)}\n{stack_trace}")
            
            # Look for 'self' argument (if it's a method) to get the operator
            if args and hasattr(args[0], 'report'):
                args[0].report({'ERROR'}, f"Error in {func.__name__}: {str(e)}")
            
            # Re-raise the exception
            raise
    
    # Copy the function's docstring and other attributes
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__module__ = func.__module__
    
    return wrapper

def validate_operator_args(func):
    """Decorator for validating operator arguments.
    
    Args:
        func (callable): Function to decorate
        
    Returns:
        callable: Decorated function
    """
    def wrapper(self, context, *args, **kwargs):
        # Validate context
        if not validate_context(context, ['scene']):
            self.report({'ERROR'}, "Invalid context")
            return {'CANCELLED'}
        
        # Get argument specs and annotations
        sig = inspect.signature(func)
        
        # Check required parameters
        for param_name, param in sig.parameters.items():
            if param.default == inspect.Parameter.empty and param_name not in ['self', 'context']:
                if param_name not in kwargs:
                    self.report({'ERROR'}, f"Missing required parameter: {param_name}")
                    return {'CANCELLED'}
        
        # Validate parameters with type annotations
        for param_name, param in sig.parameters.items():
            if param_name in kwargs and param.annotation != inspect.Parameter.empty:
                value = kwargs[param_name]
                
                # Check type
                if not isinstance(value, param.annotation):
                    self.report({'ERROR'}, f"Parameter {param_name} must be of type {param.annotation.__name__}")
                    return {'CANCELLED'}
                
                # Additional validations based on parameter name
                if "color" in param_name.lower():
                    valid, msg = validate_color(value)
                    if not valid:
                        self.report({'ERROR'}, f"Invalid color parameter {param_name}: {msg}")
                        return {'CANCELLED'}
                
                elif "matrix" in param_name.lower():
                    valid, msg = validate_matrix(value)
                    if not valid:
                        self.report({'ERROR'}, f"Invalid matrix parameter {param_name}: {msg}")
                        return {'CANCELLED'}
                
                elif "vector" in param_name.lower():
                    valid, msg = validate_vector(value)
                    if not valid:
                        self.report({'ERROR'}, f"Invalid vector parameter {param_name}: {msg}")
                        return {'CANCELLED'}
                
                elif "expression" in param_name.lower():
                    valid, msg = validate_expression(value)
                    if not valid:
                        self.report({'ERROR'}, f"Invalid expression parameter {param_name}: {msg}")
                        return {'CANCELLED'}
        
        # Execute the function
        return func(self, context, *args, **kwargs)
    
    # Copy the function's docstring and other attributes
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    wrapper.__module__ = func.__module__
    
    return wrapper

# ----------------------------------------
# Error Reporting Functions
# ----------------------------------------

def report_error(context, message, level='ERROR'):
    """Report an error to the user.
    
    Args:
        context (bpy.types.Context): Current context
        message (str): Error message
        level (str): Error level ('ERROR', 'WARNING', 'INFO')
    """
    # Log the error
    if level == 'ERROR':
        logger.error(message)
    elif level == 'WARNING':
        logger.warning(message)
    else:
        logger.info(message)
    
    # Report to Blender if in the context of an operator
    for region in context.area.regions:
        if region.type == "HEADER":
            region.tag_redraw()
    
    # Flash the error message in the Blender UI
    if hasattr(context, 'window_manager'):
        def draw_handler(self, context):
            self.layout.label(text=message)
        
        context.window_manager.popup_menu(draw_handler, title="Error", icon='ERROR')

def report_validation_errors(operator, errors):
    """Report validation errors to the user.
    
    Args:
        operator (bpy.types.Operator): Operator to report errors
        errors (list): List of error messages
        
    Returns:
        set: {'CANCELLED'} if there are errors, {'PASS_THROUGH'} otherwise
    """
    if errors:
        for error in errors:
            operator.report({'ERROR'}, error)
        return {'CANCELLED'}
    
    return {'PASS_THROUGH'}

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register error utilities"""
    logger.info("Math Playground: Error utilities registered")

def unregister():
    """Unregister error utilities"""
    logger.info("Math Playground: Error utilities unregistered")