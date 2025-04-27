# algorithms/complex.py - Complex analysis algorithms

import numpy as np
import cmath


def domain_coloring(
        func,
        x_range,
        y_range,
        resolution=100,
        progress_callback=None):
    """Generate domain coloring visualization of a complex function.

    Args:
        func (callable): Complex function to visualize
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
                if not progress_callback(
                        progress,
                        f"Evaluating function at {i*resolution+j}/{resolution*resolution} points"):
                    return None, None  # Cancelled

            try:
                W[i, j] = func(Z[i, j])
            except (ValueError, ZeroDivisionError, OverflowError):
                W[i, j] = float('nan')

    # Convert to HSV visualization
    hsv = np.zeros((resolution, resolution, 3), dtype=float)

    # Hue from argument
    hsv[:, :, 0] = (np.angle(W) / (2 * np.pi)) % 1.0

    # Saturation constant
    hsv[:, :, 1] = 1.0

    # Value from magnitude
    mag = np.abs(W)
    hsv[:, :, 2] = 1.0 - 1.0 / (1.0 + mag)

    # Convert to RGB
    rgb = np.zeros((resolution, resolution, 3), dtype=float)

    # Simple HSV to RGB conversion
    for i in range(resolution):
        for j in range(resolution):
            h, s, v = hsv[i, j]

            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c

            if h < 1 / 6:
                rgb[i, j] = [c + m, x + m, m]
            elif h < 2 / 6:
                rgb[i, j] = [x + m, c + m, m]
            elif h < 3 / 6:
                rgb[i, j] = [m, c + m, x + m]
            elif h < 4 / 6:
                rgb[i, j] = [m, x + m, c + m]
            elif h < 5 / 6:
                rgb[i, j] = [x + m, m, c + m]
            else:
                rgb[i, j] = [c + m, m, x + m]

    return rgb, hsv


def riemann_sphere_projection(z):
    """Project a complex number onto the Riemann sphere.

    Args:
        z (complex): Complex number to project

    Returns:
        tuple: (x, y, z) coordinates on the Riemann sphere
    """
    # Convert to stereographic projection
    if z == float('inf'):
        return (0, 0, 1)  # North pole

    x = z.real
    y = z.imag
    norm_squared = x * x + y * y

    # Compute 3D coordinates
    sphere_x = 2 * x / (1 + norm_squared)
    sphere_y = 2 * y / (1 + norm_squared)
    sphere_z = (norm_squared - 1) / (norm_squared + 1)

    return (sphere_x, sphere_y, sphere_z)


def inverse_riemann_sphere_projection(x, y, z):
    """Convert a point on the Riemann sphere back to a complex number.

    Args:
        x (float): X coordinate on the sphere
        y (float): Y coordinate on the sphere
        z (float): Z coordinate on the sphere

    Returns:
        complex: Corresponding complex number
    """
    # Check for north pole (infinity)
    if z >= 1.0 - 1e-10:
        return float('inf')

    # Inverse stereographic projection
    denom = 1 - z
    real_part = x / denom
    imag_part = y / denom

    return complex(real_part, imag_part)


def complex_function_iteration(func, z0, num_iterations, escape_radius=None):
    """Iterate a complex function starting from a seed point.

    Args:
        func (callable): Complex function to iterate
        z0 (complex): Initial value
        num_iterations (int): Number of iterations
        escape_radius (float, optional): Escape radius for divergence detection

    Returns:
        list: Sequence of complex values from iteration
    """
    sequence = [z0]
    z = z0

    for i in range(num_iterations):
        try:
            z = func(z)

            # Check for divergence if escape radius is specified
            if escape_radius and abs(z) > escape_radius:
                break

            sequence.append(z)

            # Check for convergence
            if abs(z - sequence[-2]) < 1e-10:
                break

        except (ValueError, ZeroDivisionError, OverflowError):
            break

    return sequence


def mobius_transformation(z, a, b, c, d):
    """Apply Möbius transformation to a complex number.

    Args:
        z (complex): Complex number to transform
        a (complex): Parameter a
        b (complex): Parameter b
        c (complex): Parameter c
        d (complex): Parameter d

    Returns:
        complex: Transformed complex number
    """
    # Check determinant is non-zero
    if abs(a * d - b * c) < 1e-10:
        raise ValueError("Invalid Möbius transformation (ad-bc ≈ 0)")

    # Handle infinity cases
    if z == float('inf'):
        if c == 0:
            return float('inf')
        else:
            return a / c

    if c * z + d == 0:
        return float('inf')

    return (a * z + b) / (c * z + d)

# Registration functions


def register():
    """Register complex analysis algorithms"""
    print("Math Playground: Complex analysis algorithms registered")


def unregister():
    """Unregister complex analysis algorithms"""
    print("Math Playground: Complex analysis algorithms unregistered")
