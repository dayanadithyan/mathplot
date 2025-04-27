# utils/math_utils.py - Mathematical utilities for Math Playground

import numpy as np
import math
from mathutils import Vector, Matrix

# ----------------------------------------
# Safe Expression Evaluation
# ----------------------------------------

def evaluate_expression(expression, variables=None):
    """Safely evaluate a mathematical expression.
    
    Args:
        expression (str): Mathematical expression as string
        variables (dict): Dictionary of variables to use in evaluation
        
    Returns:
        float: Result of expression evaluation
        
    Raises:
        ValueError: If expression is invalid or unsafe
    """
    if variables is None:
        variables = {}
    
    # Provide a safe namespace for evaluation
    safe_namespace = {
        "math": math,
        "np": np,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "pow": pow,
        "min": min,
        "max": max,
        **variables
    }
    
    # Check for potentially unsafe operations
    unsafe_terms = ["__", "import ", "eval(", "exec(", "compile(", "globals(", "locals(", 
                   "getattr(", "setattr(", "delattr(", "open(", "file(", "os.", "sys."]
    for term in unsafe_terms:
        if term in expression:
            raise ValueError(f"Unsafe term detected in expression: {term}")
    
    try:
        return eval(expression, {"__builtins__": {}}, safe_namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")

def evaluate_vector_expression(expression, variables=None, dimensions=3):
    """Evaluate an expression that should return a vector.
    
    Args:
        expression (str): Mathematical expression as string
        variables (dict): Dictionary of variables to use in evaluation
        dimensions (int): Expected number of dimensions
        
    Returns:
        numpy.ndarray: Result as a vector
        
    Raises:
        ValueError: If expression is invalid or doesn't return a vector
    """
    result = evaluate_expression(expression, variables)
    
    # Convert to numpy array if not already
    if not isinstance(result, (np.ndarray, list, tuple, Vector)):
        raise ValueError(f"Expression must return a vector, got {type(result)}")
    
    # Convert to numpy array
    if isinstance(result, Vector):
        result = np.array([result.x, result.y, result.z])
    else:
        result = np.array(result)
    
    # Check dimensions
    if result.size != dimensions:
        raise ValueError(f"Expression must return a {dimensions}D vector, got {result.size}D")
    
    return result

def evaluate_matrix_expression(expression, variables=None, shape=(3, 3)):
    """Evaluate an expression that should return a matrix.
    
    Args:
        expression (str): Mathematical expression as string
        variables (dict): Dictionary of variables to use in evaluation
        shape (tuple): Expected shape of the matrix
        
    Returns:
        numpy.ndarray: Result as a matrix
        
    Raises:
        ValueError: If expression is invalid or doesn't return a matrix
    """
    result = evaluate_expression(expression, variables)
    
    # Convert to numpy array if not already
    if not isinstance(result, (np.ndarray, list, tuple, Matrix)):
        raise ValueError(f"Expression must return a matrix, got {type(result)}")
    
    # Convert to numpy array
    if isinstance(result, Matrix):
        result = np.array(result)
    else:
        result = np.array(result)
    
    # Check dimensions
    if result.shape != shape:
        raise ValueError(f"Expression must return a {shape} matrix, got {result.shape}")
    
    return result

# ----------------------------------------
# Number Theory Functions
# ----------------------------------------

def is_prime(n):
    """Determine if a number is prime.
    
    Args:
        n (int): Number to check
        
    Returns:
        bool: True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def generate_primes(limit, progress_callback=None):
    """Generate a list of prime numbers up to a limit.
    
    Args:
        limit (int): Upper limit for prime generation
        progress_callback (callable): Function to report progress
        
    Returns:
        list: List of prime numbers
    """
    primes = []
    for i in range(2, limit + 1):
        # Report progress
        if progress_callback and i % 100 == 0:
            progress = i / limit
            if not progress_callback(progress, f"Checking {i}/{limit}"):
                return primes  # Cancelled
        
        if is_prime(i):
            primes.append(i)
    
    return primes

def generate_sieve_of_eratosthenes(limit, progress_callback=None):
    """Generate prime numbers using the Sieve of Eratosthenes algorithm.
    
    Args:
        limit (int): Upper limit for prime generation
        progress_callback (callable): Function to report progress
        
    Returns:
        list: List of prime numbers
    """
    # Initialize sieve
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    # Mark non-primes
    for i in range(2, int(math.sqrt(limit)) + 1):
        # Report progress
        if progress_callback and i % 100 == 0:
            progress = i / math.sqrt(limit)
            if not progress_callback(progress, f"Sieving {i}/{int(math.sqrt(limit))}"):
                return []  # Cancelled
        
        if sieve[i]:
            # Mark multiples of i as non-prime
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    # Collect primes
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    return primes

def prime_factorization(n, progress_callback=None):
    """Find the prime factorization of a number.
    
    Args:
        n (int): Number to factorize
        progress_callback (callable): Function to report progress
        
    Returns:
        dict: Dictionary mapping prime factors to their exponents
    """
    factors = {}
    d = 2
    
    while d * d <= n:
        # Report progress
        if progress_callback:
            progress = (d * d) / n
            if not progress_callback(progress, f"Factorizing: Testing {d}"):
                return factors  # Cancelled
        
        while n % d == 0:
            if d in factors:
                factors[d] += 1
            else:
                factors[d] = 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = 1
    
    return factors

def generate_sequence(sequence_type, length, formula=None, progress_callback=None):
    """Generate an integer sequence.
    
    Args:
        sequence_type (str): Type of sequence ('FIBONACCI', 'SQUARE', 'TRIANGULAR', 'PRIME', 'CUSTOM')
        length (int): Number of terms to generate
        formula (str, optional): Custom formula for 'CUSTOM' type
        progress_callback (callable): Function to report progress
        
    Returns:
        list: The generated sequence
    """
    sequence = []
    
    if sequence_type == 'FIBONACCI':
        # Fibonacci sequence
        a, b = 0, 1
        for i in range(length):
            if progress_callback:
                progress = (i + 1) / length
                if not progress_callback(progress, f"Generating term {i+1}/{length}"):
                    return sequence  # Cancelled
            
            sequence.append(a)
            a, b = b, a + b
    
    elif sequence_type == 'SQUARE':
        # Square numbers
        for i in range(1, length + 1):
            if progress_callback:
                progress = i / length
                if not progress_callback(progress, f"Generating term {i}/{length}"):
                    return sequence  # Cancelled
            
            sequence.append(i**2)
    
    elif sequence_type == 'TRIANGULAR':
        # Triangular numbers
        for i in range(1, length + 1):
            if progress_callback:
                progress = i / length
                if not progress_callback(progress, f"Generating term {i}/{length}"):
                    return sequence  # Cancelled
            
            sequence.append(i * (i + 1) // 2)
    
    elif sequence_type == 'PRIME':
        # Prime numbers
        primes = generate_primes(length * 10, progress_callback)
        sequence = primes[:length]
    
    elif sequence_type == 'CUSTOM':
        # Custom formula
        for i in range(1, length + 1):
            if progress_callback:
                progress = i / length
                if not progress_callback(progress, f"Generating term {i}/{length}"):
                    return sequence  # Cancelled
            
            try:
                value = evaluate_expression(formula, {"n": i})
                sequence.append(value)
            except ValueError as e:
                raise ValueError(f"Error in custom formula at term {i}: {e}")
    
    return sequence

# ----------------------------------------
# Linear Algebra Functions
# ----------------------------------------

def parse_matrix(matrix_str):
    """Parse a matrix from a string representation.
    
    Args:
        matrix_str (str): Matrix in format 'a,b,c;d,e,f;g,h,i'
        
    Returns:
        numpy.ndarray: The parsed matrix
        
    Raises:
        ValueError: If matrix format is invalid
    """
    try:
        # Split into rows
        rows = matrix_str.split(';')
        
        # Parse each row
        matrix = []
        for row in rows:
            values = row.split(',')
            matrix.append([float(v.strip()) for v in values])
        
        # Check that matrix is rectangular
        row_lengths = [len(row) for row in matrix]
        if len(set(row_lengths)) != 1:
            raise ValueError(f"Matrix rows have inconsistent lengths: {row_lengths}")
        
        return np.array(matrix)
    
    except Exception as e:
        raise ValueError(f"Invalid matrix format: {e}")

def apply_transformation(vectors, matrix):
    """Apply a transformation matrix to a list of vectors.
    
    Args:
        vectors (list): List of vectors to transform
        matrix (numpy.ndarray): Transformation matrix
        
    Returns:
        list: Transformed vectors
    """
    # Convert matrix to numpy if needed
    if isinstance(matrix, Matrix):
        matrix = np.array(matrix)
    
    # Apply transformation to each vector
    transformed = []
    for vec in vectors:
        # Convert to numpy if needed
        if isinstance(vec, Vector):
            vec = np.array([vec.x, vec.y, vec.z])
        
        # Apply transformation
        result = matrix @ vec
        
        # Convert back to original type
        if isinstance(vec, Vector):
            result = Vector((result[0], result[1], result[2]))
        
        transformed.append(result)
    
    return transformed

def compute_determinant(matrix):
    """Compute the determinant of a matrix.
    
    Args:
        matrix (numpy.ndarray or mathutils.Matrix): Input matrix
        
    Returns:
        float: Determinant of the matrix
    """
    if isinstance(matrix, Matrix):
        return matrix.determinant
    else:
        return np.linalg.det(matrix)

def compute_inverse(matrix):
    """Compute the inverse of a matrix.
    
    Args:
        matrix (numpy.ndarray or mathutils.Matrix): Input matrix
        
    Returns:
        numpy.ndarray or mathutils.Matrix: Inverse of the matrix
        
    Raises:
        ValueError: If matrix is singular
    """
    if isinstance(matrix, Matrix):
        try:
            return matrix.inverted()
        except ValueError:
            raise ValueError("Matrix is singular and cannot be inverted")
    else:
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is singular and cannot be inverted")

def compute_eigenvalues(matrix):
    """Compute the eigenvalues of a matrix.
    
    Args:
        matrix (numpy.ndarray or mathutils.Matrix): Input matrix
        
    Returns:
        numpy.ndarray: Eigenvalues of the matrix
    """
    if isinstance(matrix, Matrix):
        matrix = np.array(matrix)
    
    return np.linalg.eigvals(matrix)

def compute_eigenvectors(matrix):
    """Compute the eigenvectors of a matrix.
    
    Args:
        matrix (numpy.ndarray or mathutils.Matrix): Input matrix
        
    Returns:
        tuple: (eigenvalues, eigenvectors)
    """
    if isinstance(matrix, Matrix):
        matrix = np.array(matrix)
    
    return np.linalg.eig(matrix)

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register mathematical utilities"""
    print("Math Playground: Mathematical utilities registered")

def unregister():
    """Unregister mathematical utilities"""
    print("Math Playground: Mathematical utilities unregistered")