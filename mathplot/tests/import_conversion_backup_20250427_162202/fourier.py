# algorithms/fourier.py - Fourier series algorithms

import numpy as np

def compute_fourier_coefficients(func, period, n_terms):
    """Compute Fourier series coefficients for a function.
    
    Args:
        func (callable): Function to approximate
        period (float): Period of the function
        n_terms (int): Number of terms to compute
        
    Returns:
        tuple: (a0, a_n, b_n) - Constant term and coefficient arrays
    """
    # Angular frequency
    omega = 2 * np.pi / period
    
    # Compute a0 (constant term)
    a0 = 2 / period * np.trapz([func(x) for x in np.linspace(0, period, 1000)], 
                               dx=period/1000)
    
    # Compute a_n and b_n coefficients
    a_n = np.zeros(n_terms)
    b_n = np.zeros(n_terms)
    
    x = np.linspace(0, period, 1000)
    
    for n in range(1, n_terms + 1):
        # a_n coefficient
        integrand_a = np.array([func(t) * np.cos(n * omega * t) for t in x])
        a_n[n-1] = 2 / period * np.trapz(integrand_a, dx=period/1000)
        
        # b_n coefficient
        integrand_b = np.array([func(t) * np.sin(n * omega * t) for t in x])
        b_n[n-1] = 2 / period * np.trapz(integrand_b, dx=period/1000)
    
    return a0, a_n, b_n

def evaluate_fourier_series(x, a0, a_n, b_n, period):
    """Evaluate a Fourier series at point x.
    
    Args:
        x (float or array): Point(s) to evaluate
        a0 (float): Constant term
        a_n (array): Cosine coefficients
        b_n (array): Sine coefficients
        period (float): Period of the function
        
    Returns:
        float or array: Function value(s) at x
    """
    # Angular frequency
    omega = 2 * np.pi / period
    
    # Start with constant term
    result = a0 / 2
    
    # Add harmonic terms
    for n in range(1, len(a_n) + 1):
        result += a_n[n-1] * np.cos(n * omega * x) + b_n[n-1] * np.sin(n * omega * x)
    
    return result

def generate_fourier_components(a0, a_n, b_n, period, x_values):
    """Generate individual Fourier components.
    
    Args:
        a0 (float): Constant term
        a_n (array): Cosine coefficients
        b_n (array): Sine coefficients
        period (float): Period of the function
        x_values (array): X values for evaluation
        
    Returns:
        list: List of component functions evaluated at x_values
    """
    # Angular frequency
    omega = 2 * np.pi / period
    
    components = []
    
    # Constant term
    components.append(np.ones_like(x_values) * a0 / 2)
    
    # Harmonic terms
    for n in range(1, len(a_n) + 1):
        # Cosine component
        cos_component = a_n[n-1] * np.cos(n * omega * x_values)
        components.append(cos_component)
        
        # Sine component
        sin_component = b_n[n-1] * np.sin(n * omega * x_values)
        components.append(sin_component)
    
    return components

def function_to_fourier(func, period, n_terms, x_values):
    """Convert a function to its Fourier series representation.
    
    Args:
        func (callable): Function to approximate
        period (float): Period of the function
        n_terms (int): Number of terms to compute
        x_values (array): X values for evaluation
        
    Returns:
        tuple: (series_values, components)
    """
    # Compute coefficients
    a0, a_n, b_n = compute_fourier_coefficients(func, period, n_terms)
    
    # Evaluate series
    series_values = evaluate_fourier_series(x_values, a0, a_n, b_n, period)
    
    # Generate components
    components = generate_fourier_components(a0, a_n, b_n, period, x_values)
    
    return series_values, components

# Registration functions
def register():
    """Register Fourier series algorithms"""
    print("Math Playground: Fourier series algorithms registered")

def unregister():
    """Unregister Fourier series algorithms"""
    print("Math Playground: Fourier series algorithms unregistered")