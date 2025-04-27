# mathplot/utils/error_utils.py (continued)

# ----------------------------------------
# Function Validation
# ----------------------------------------

def validate_function_args(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to validate function arguments based on type annotations.
    
    Args:
        func: The function to decorate
    
    Returns:
        The decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function signature
        sig = inspect.signature(func)
        
        # Get bound arguments
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
        except TypeError as e:
            raise ValueError(f"Invalid arguments: {str(e)}")
        
        # Check each parameter
        for param_name, param_value in bound_args.arguments.items():
            param = sig.parameters.get(param_name)
            if param is None:
                continue
            
            # Get annotation if available
            annotation = param.annotation
            if annotation is inspect.Parameter.empty:
                continue
            
            # Check if value matches annotation
            if not isinstance(param_value, annotation) and annotation is not Any:
                raise TypeError(f"Parameter '{param_name}' must be of type {annotation.__name__}, got {type(param_value).__name__}")
        
        # All validations passed, execute the function
        return func(*args, **kwargs)
    
    return wrapper

# ----------------------------------------
# Blender Object Validation
# ----------------------------------------

def ensure_valid_object(obj: Optional[bpy.types.Object], 
                      expected_type: Optional[str] = None) -> Tuple[bool, str]:
    """Validate that an object exists and is of the expected type.
    
    Args:
        obj: The object to validate
        expected_type: Expected object type (e.g., 'MESH', 'CURVE')
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    if obj is None:
        return False, "Object is None"
    
    if not isinstance(obj, bpy.types.Object):
        return False, "Not a valid Blender object"
    
    if expected_type and obj.type != expected_type:
        return False, f"Object is of type '{obj.type}', expected '{expected_type}'"
    
    return True, ""

def validate_mesh(mesh: Any) -> Tuple[bool, str]:
    """Validate a Blender mesh datablock.
    
    Args:
        mesh: The mesh to validate
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    if mesh is None:
        return False, "Mesh is None"
    
    if not isinstance(mesh, bpy.types.Mesh):
        return False, "Not a valid Blender mesh"
    
    # Check for basic mesh integrity
    if not mesh.vertices or len(mesh.vertices) == 0:
        return False, "Mesh has no vertices"
    
    return True, ""

# ----------------------------------------
# Math Validation
# ----------------------------------------

def validate_sequence(sequence_type: str, 
                    length: int, 
                    formula: Optional[str] = None) -> Tuple[bool, str]:
    """Validate parameters for sequence generation.
    
    Args:
        sequence_type: Type of sequence
        length: Number of terms to generate
        formula: Custom formula for custom sequences
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Validate sequence type
    valid_types = ['FIBONACCI', 'SQUARE', 'TRIANGULAR', 'PRIME', 'CUSTOM']
    if sequence_type not in valid_types:
        return False, f"Invalid sequence type, must be one of {valid_types}"
    
    # Validate length
    if length < 1:
        return False, "Sequence length must be at least 1"
    
    # Validate formula for custom sequences
    if sequence_type == 'CUSTOM' and not formula:
        return False, "Custom formula is required for custom sequences"
    
    # Validate custom formula if provided
    if formula:
        return validate_expression(formula, ['n'], {'n': 1})
    
    return True, ""

def validate_graph_params(node_count: int, 
                        edge_probability: float) -> Tuple[bool, str]:
    """Validate parameters for graph generation.
    
    Args:
        node_count: Number of nodes
        edge_probability: Probability of edges between nodes
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Validate node count
    if node_count < 2:
        return False, "Graph must have at least 2 nodes"
    
    # Validate edge probability
    if edge_probability < 0 or edge_probability > 1:
        return False, "Edge probability must be between 0 and 1"
    
    return True, ""

def validate_function_params(function: str, 
                           x_min: float, 
                           x_max: float, 
                           samples: int) -> Tuple[bool, str]:
    """Validate parameters for function plotting.
    
    Args:
        function: Function expression
        x_min: Minimum x value
        x_max: Maximum x value
        samples: Number of samples
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Validate function expression
    valid, message = validate_expression(function, ['x'], {'x': x_min})
    if not valid:
        return False, message
    
    # Validate range
    if x_min >= x_max:
        return False, "x_min must be less than x_max"
    
    # Validate samples
    if samples < 10:
        return False, "Must have at least 10 samples"
    
    return True, ""

def validate_vector_field_params(x_component: str, 
                               y_component: str, 
                               z_component: str) -> Tuple[bool, str]:
    """Validate parameters for vector field plotting.
    
    Args:
        x_component: X component expression
        y_component: Y component expression
        z_component: Z component expression
    
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Test variables
    test_vars = {'x': 0, 'y': 0, 'z': 0}
    
    # Validate x component
    valid, message = validate_expression(x_component, ['x', 'y', 'z'], test_vars)
    if not valid:
        return False, f"Invalid x component: {message}"
    
    # Validate y component
    valid, message = validate_expression(y_component, ['x', 'y', 'z'], test_vars)
    if not valid:
        return False, f"Invalid y component: {message}"
    
    # Validate z component
    valid, message = validate_expression(z_component, ['x', 'y', 'z'], test_vars)
    if not valid:
        return False, f"Invalid z component: {message}"
    
    return True, ""

# ----------------------------------------
# Error Handling Helper Functions
# ----------------------------------------

def handle_operator_error(self, error: Exception, context: bpy.types.Context) -> Set[str]:
    """Handle an error in an operator and return appropriate return set.
    
    Args:
        self: Operator instance
        error: The exception that occurred
        context: Current context
    
    Returns:
        Operator return set (e.g., {'CANCELLED'})
    """
    # Log the error
    print(f"Error in {self.bl_idname}:")
    traceback.print_exc()
    
    # Report the error to the user
    self.report({'ERROR'}, f"Error: {str(error)}")
    
    # End progress reporting if it was started
    try:
        from mathplot.utils.progress import end_progress
        end_progress(context)
    except:
        # Fallback if the progress module is not available
        if hasattr(context.window_manager, "progress_end"):
            context.window_manager.progress_end()
    
    return {'CANCELLED'}

def summarize_validation_errors(errors: List[Tuple[str, str]]) -> str:
    """Summarize multiple validation errors into a single message.
    
    Args:
        errors: List of (field_name, error_message) tuples
    
    Returns:
        Summary error message
    """
    if not errors:
        return ""
    
    if len(errors) == 1:
        return f"{errors[0][0]}: {errors[0][1]}"
    
    summary = "Multiple validation errors:"
    for field, message in errors:
        summary += f"\n- {field}: {message}"
    
    return summary

def validate_and_report(self, validations: List[Tuple[bool, str, str]]) -> bool:
    """Validate multiple conditions and report errors.
    
    Args:
        self: Operator instance
        validations: List of (is_valid, error_message, field_name) tuples
    
    Returns:
        Whether all validations passed
    """
    errors = []
    
    for is_valid, message, field in validations:
        if not is_valid:
            errors.append((field, message))
    
    if errors:
        self.report({'ERROR'}, summarize_validation_errors(errors))
        return False
    
    return True

def wrap_exception_with_context(func_name: str) -> Callable[[Exception], Exception]:
    """Create a wrapper for re-raising exceptions with additional context.
    
    Args:
        func_name: Name of the function for context
    
    Returns:
        Function to wrap exceptions
    """
    def wrapper(exception: Exception) -> Exception:
        message = f"Error in {func_name}: {str(exception)}"
        return type(exception)(message)
    
    return wrapper