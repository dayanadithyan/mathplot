# algorithms/differential.py - Differential equation algorithms

import numpy as np

def euler_method(func, x0, y0, x_end, h):
    """Implement Euler method for solving ODEs.
    
    Args:
        func (callable): Function f(x, y) in the ODE dy/dx = f(x, y)
        x0 (float): Initial x value
        y0 (float): Initial y value
        x_end (float): End x value
        h (float): Step size
        
    Returns:
        tuple: (x_values, y_values) - Arrays of x and y values
    """
    # Number of steps
    n_steps = int((x_end - x0) / h)
    
    # Arrays to store results
    x_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set initial values
    x_values[0] = x0
    y_values[0] = y0
    
    # Iterate through steps
    for i in range(n_steps):
        x = x_values[i]
        y = y_values[i]
        
        # Euler step
        y_values[i+1] = y + h * func(x, y)
        x_values[i+1] = x + h
    
    return x_values, y_values

def runge_kutta_4(func, x0, y0, x_end, h):
    """Implement 4th-order Runge-Kutta method for solving ODEs.
    
    Args:
        func (callable): Function f(x, y) in the ODE dy/dx = f(x, y)
        x0 (float): Initial x value
        y0 (float): Initial y value
        x_end (float): End x value
        h (float): Step size
        
    Returns:
        tuple: (x_values, y_values) - Arrays of x and y values
    """
    # Number of steps
    n_steps = int((x_end - x0) / h)
    
    # Arrays to store results
    x_values = np.zeros(n_steps + 1)
    y_values = np.zeros(n_steps + 1)
    
    # Set initial values
    x_values[0] = x0
    y_values[0] = y0
    
    # Iterate through steps
    for i in range(n_steps):
        x = x_values[i]
        y = y_values[i]
        
        # Calculate k values
        k1 = h * func(x, y)
        k2 = h * func(x + h/2, y + k1/2)
        k3 = h * func(x + h/2, y + k2/2)
        k4 = h * func(x + h, y + k3)
        
        # Update y and x
        y_values[i+1] = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x_values[i+1] = x + h
    
    return x_values, y_values

def adaptive_step_size(func, x0, y0, x_end, h0, tol=1e-6):
    """Implement adaptive step size method for solving ODEs.
    
    Args:
        func (callable): Function f(x, y) in the ODE dy/dx = f(x, y)
        x0 (float): Initial x value
        y0 (float): Initial y value
        x_end (float): End x value
        h0 (float): Initial step size
        tol (float): Error tolerance
        
    Returns:
        tuple: (x_values, y_values) - Arrays of x and y values
    """
    # Lists to store results
    x_values = [x0]
    y_values = [y0]
    
    # Current position
    x = x0
    y = y0
    h = h0
    
    while x < x_end:
        # Adjust step size to not exceed x_end
        if x + h > x_end:
            h = x_end - x
        
        # Calculate two approximations
        # Using step h
        k1 = h * func(x, y)
        k2 = h * func(x + h/2, y + k1/2)
        k3 = h * func(x + h/2, y + k2/2)
        k4 = h * func(x + h, y + k3)
        y_h = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        
        # Using two steps of h/2
        h_half = h / 2
        # First half step
        k1_half = h_half * func(x, y)
        k2_half = h_half * func(x + h_half/2, y + k1_half/2)
        k3_half = h_half * func(x + h_half/2, y + k2_half/2)
        k4_half = h_half * func(x + h_half, y + k3_half)
        y_half = y + (k1_half + 2*k2_half + 2*k3_half + k4_half) / 6
        
        # Second half step
        x_half = x + h_half
        k1_half2 = h_half * func(x_half, y_half)
        k2_half2 = h_half * func(x_half + h_half/2, y_half + k1_half2/2)
        k3_half2 = h_half * func(x_half + h_half/2, y_half + k2_half2/2)
        k4_half2 = h_half * func(x_half + h_half, y_half + k3_half2)
        y_h2 = y_half + (k1_half2 + 2*k2_half2 + 2*k3_half2 + k4_half2) / 6
        
        # Estimate error
        error = abs(y_h - y_h2)
        
        # Adjust step size
        if error < tol:
            # Step successful, update values
            x = x + h
            y = y_h2  # Use more accurate approximation
            x_values.append(x)
            y_values.append(y)
            
            # Increase step size
            h = h * min(2, max(0.5, 0.9 * (tol / max(error, 1e-10))**0.2))
        else:
            # Reduce step size and retry
            h = h * max(0.1, 0.9 * (tol / max(error, 1e-10))**0.25)
    
    return np.array(x_values), np.array(y_values)

# Registration functions
def register():
    """Register differential equation algorithms"""
    print("Math Playground: Differential equation algorithms registered")

def unregister():
    """Unregister differential equation algorithms"""
    print("Math Playground: Differential equation algorithms unregistered")