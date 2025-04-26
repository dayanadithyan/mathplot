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
# Calculus Functions
# ----------------------------------------

def numerical_derivative(func, x, h=1e-5):
    """Compute the numerical derivative of a function at a point.
    
    Args:
        func (callable): Function to differentiate
        x (float): Point at which to compute the derivative
        h (float): Step size
        
    Returns:
        float: Derivative of func at x
    """
    return (func(x + h) - func(x - h)) / (2 * h)

def numerical_integral(func, a, b, n=1000):
    """Compute the numerical integral of a function over an interval.
    
    Args:
        func (callable): Function to integrate
        a (float): Lower bound
        b (float): Upper bound
        n (int): Number of intervals
        
    Returns:
        float: Integral of func from a to b
    """
    # Trapezoidal rule
    h = (b - a) / n
    result = 0.5 * (func(a) + func(b))
    
    for i in range(1, n):
        result += func(a + i * h)
    
    result *= h
    return result

def taylor_series(func, x0, x, n=5):
    """Compute the Taylor series approximation of a function.
    
    Args:
        func (callable): Function to approximate
        x0 (float): Expansion point
        x (float): Evaluation point
        n (int): Number of terms
        
    Returns:
        float: Taylor series approximation of func(x)
    """
    result = func(x0)
    factorial = 1
    h = x - x0
    
    for i in range(1, n + 1):
        # Compute derivative
        derivative = numerical_derivative(
            lambda t: numerical_derivative(func, t, h=1e-5), 
            x0, 
            h=1e-5
        ) if i > 1 else numerical_derivative(func, x0, h=1e-5)
        
        # Add term
        factorial *= i
        result += derivative * (h ** i) / factorial
    
    return result

# ----------------------------------------
# Complex Analysis Functions
# ----------------------------------------

def complex_function(func_str, z, variables=None):
    """Evaluate a complex function.
    
    Args:
        func_str (str): Function expression as string
        z (complex): Complex input
        variables (dict, optional): Additional variables
        
    Returns:
        complex: Result of the function
    """
    if variables is None:
        variables = {}
    
    # Add complex number to variables
    variables['z'] = z
    
    # Add complex functions to namespace
    safe_namespace = {
        "math": math,
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": abs,
        "conj": np.conj,
        "real": np.real,
        "imag": np.imag,
        "phase": np.angle,
        **variables
    }
    
    # Check for unsafe operations
    unsafe_terms = ["__", "import ", "eval(", "exec(", "compile(", "globals(", "locals("]
    for term in unsafe_terms:
        if term in func_str:
            raise ValueError(f"Unsafe term detected in expression: {term}")
    
    try:
        return eval(func_str, {"__builtins__": {}}, safe_namespace)
    except Exception as e:
        raise ValueError(f"Error evaluating complex function: {e}")

def domain_coloring(func_str, x_range, y_range, resolution=100, progress_callback=None):
    """Generate domain coloring visualization of a complex function.
    
    Args:
        func_str (str): Function expression as string
        x_range (tuple): Range of real component (min, max)
        y_range (tuple): Range of imaginary component (min, max)
        resolution (int): Grid resolution
        progress_callback (callable): Function to report progress
        
    Returns:
        tuple: (rgb_array, hsv_array) - RGB and HSV visualizations
    """
    # Create grid
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    # Evaluate function at each point
    W = np.zeros_like(Z, dtype=complex)
    
    for i in range(resolution):
        for j in range(resolution):
            # Report progress
            if progress_callback:
                progress = (i * resolution + j) / (resolution * resolution)
                if not progress_callback(progress, f"Evaluating function at {i*resolution+j}/{resolution*resolution} points"):
                    return None, None  # Cancelled
            
            try:
                W[i, j] = complex_function(func_str, Z[i, j])
            except ValueError:
                W[i, j] = float('nan')
    
    # Convert to HSV visualization
    hsv = np.zeros((resolution, resolution, 3), dtype=float)
    
    # Hue from argument
    hsv[:,:,0] = (np.angle(W) / (2 * np.pi)) % 1.0
    
    # Saturation constant
    hsv[:,:,1] = 1.0
    
    # Value from magnitude
    mag = np.abs(W)
    hsv[:,:,2] = 1.0 - 1.0 / (1.0 + mag)
    
    # Convert to RGB
    rgb = np.zeros((resolution, resolution, 3), dtype=float)
    
    # Simple HSV to RGB conversion
    for i in range(resolution):
        for j in range(resolution):
            h, s, v = hsv[i, j]
            
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c
            
            if h < 1/6:
                rgb[i, j] = [c+m, x+m, m]
            elif h < 2/6:
                rgb[i, j] = [x+m, c+m, m]
            elif h < 3/6:
                rgb[i, j] = [m, c+m, x+m]
            elif h < 4/6:
                rgb[i, j] = [m, x+m, c+m]
            elif h < 5/6:
                rgb[i, j] = [x+m, m, c+m]
            else:
                rgb[i, j] = [c+m, m, x+m]
    
    return rgb, hsv

# ----------------------------------------
# Graph Theory Functions
# ----------------------------------------

def create_graph_adjacency_list(edges, directed=False):
    """Create an adjacency list representation of a graph.
    
    Args:
        edges (list): List of (start, end, weight) tuples
        directed (bool): Whether the graph is directed
        
    Returns:
        dict: Adjacency list representation {node: [(neighbor, weight), ...]}
    """
    graph = {}
    
    for edge in edges:
        if len(edge) >= 2:
            start, end = edge[0], edge[1]
            weight = edge[2] if len(edge) >= 3 else 1.0
            
            # Add start node if not exists
            if start not in graph:
                graph[start] = []
            
            # Add end node if not exists
            if end not in graph:
                graph[end] = []
            
            # Add edge
            graph[start].append((end, weight))
            
            # Add reverse edge for undirected graphs
            if not directed:
                graph[end].append((start, weight))
    
    return graph

def shortest_path_dijkstra(graph, start, end):
    """Find shortest path between two nodes using Dijkstra's algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        start: Starting node
        end: Ending node
        
    Returns:
        tuple: (distance, path) - Total distance and node path
    """
    # Initialize
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    unvisited = set(graph.keys())
    
    while unvisited:
        # Find the unvisited node with the smallest distance
        current = min(unvisited, key=lambda x: distances[x])
        
        # If we reached the end node or if the smallest distance is infinity, stop
        if current == end or distances[current] == float('inf'):
            break
        
        # Remove the current node from unvisited
        unvisited.remove(current)
        
        # Update distances to neighbors
        for neighbor, weight in graph[current]:
            distance = distances[current] + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
    
    # Reconstruct the path
    if distances[end] == float('inf'):
        return float('inf'), []  # No path found
    
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    
    path.reverse()
    return distances[end], path

def minimum_spanning_tree_kruskal(graph):
    """Find minimum spanning tree using Kruskal's algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        list: List of (start, end, weight) edges in the MST
    """
    # Extract edges and sort by weight
    edges = set()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            if node < neighbor:  # Avoid duplicates for undirected graph
                edges.add((node, neighbor, weight))
            elif node > neighbor and (neighbor, node, weight) not in edges:
                edges.add((node, neighbor, weight))
    
    edges = sorted(edges, key=lambda x: x[2])
    
    # Initialize disjoint set data structure
    parent = {node: node for node in graph}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Run Kruskal's algorithm
    mst_edges = []
    
    for edge in edges:
        start, end, weight = edge
        if find(start) != find(end):
            union(start, end)
            mst_edges.append(edge)
    
    return mst_edges

def color_graph_greedy(graph):
    """Color a graph using a greedy algorithm.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        dict: Node colors {node: color_index}
    """
    # Sort nodes by degree (number of neighbors) for better results
    nodes = sorted(graph.keys(), key=lambda x: len(graph[x]), reverse=True)
    
    # Initialize colors
    colors = {}
    
    for node in nodes:
        # Get colors of neighbors
        neighbor_colors = set()
        for neighbor, _ in graph[node]:
            if neighbor in colors:
                neighbor_colors.add(colors[neighbor])
        
        # Find the first available color
        color = 0
        while color in neighbor_colors:
            color += 1
        
        # Assign color
        colors[node] = color
    
    return colors

def detect_cycles(graph):
    """Detect cycles in a graph using DFS.
    
    Args:
        graph (dict): Adjacency list representation {node: [(neighbor, weight), ...]}
        
    Returns:
        list: List of cycles as node lists
    """
    visited = set()
    parent = {}
    cycles = []
    
    def dfs(node, current_path):
        visited.add(node)
        current_path.add(node)
        
        for neighbor, _ in graph[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                dfs(neighbor, current_path)
            elif neighbor in current_path and parent.get(node) != neighbor:
                # Found a cycle
                cycle = []
                current = node
                while current != neighbor:
                    cycle.append(current)
                    current = parent[current]
                cycle.append(neighbor)
                cycle.append(node)  # Complete the cycle
                cycles.append(cycle)
        
        current_path.remove(node)
    
    for node in graph:
        if node not in visited:
            dfs(node, set())
    
    return cycles

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register mathematical utilities"""
    print("Math Playground: Mathematical utilities registered")

def unregister():
    """Unregister mathematical utilities"""
    print("Math Playground: Mathematical utilities unregistered")