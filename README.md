# Math Playground for Blender

A comprehensive Blender add-on for exploring mathematical concepts in 3D space. This add-on provides tools for visualizing and interacting with various mathematical structures and phenomena, making it ideal for educational purposes, mathematical exploration, or creating mathematically-based art.

## Features

### Linear Algebra

- Create and manipulate vectors in 3D space
- Apply matrix transformations
- Visualize basis vectors and transformations
- Create parametric vector trails

### Number Theory

- Generate and visualize prime numbers
- Create integer sequences (Fibonacci, square, triangular, etc.)
- Visualize patterns in number sequences

### Analysis

- Plot 2D and 3D functions
- Visualize vector fields
- Create parametric curves and surfaces

### Graph Theory

- Create graph structures with various layout algorithms
- Run graph algorithms (shortest path, MST, coloring)
- Interactive graph editing

### Differential Equations

- Solve ordinary differential equations numerically
- Visualize slope fields
- Create phase portraits

### Complex Analysis

- Domain coloring of complex functions
- Riemann sphere visualization
- Complex transformations

### Fourier Series

- Visualize Fourier series approximations of functions
- Show individual Fourier components
- Animate series convergence

## Installation

1. Download the `math_playground.zip` file from the releases section
2. In Blender, go to Edit → Preferences → Add-ons
3. Click "Install..." and select the downloaded zip file
4. Enable the add-on by checking the box next to "3D View: Math Playground"

## Usage

After installation, Math Playground can be accessed from the 3D View sidebar. Press `N` in the 3D View to open the sidebar if it isn't already visible, then select the "Math Playground" tab.

The interface is organized by mathematical domains:

1. Select a module from the main panel
2. Adjust the settings specific to that module
3. Create mathematical objects using the operators provided
4. Use the "Clear" buttons to remove objects when finished

### Example: Creating a Vector Field

1. Select the "Analysis" module
2. In the vector field section, enter expressions for the X, Y, and Z components
   - For example: X: `-y`, Y: `x`, Z: `0` creates a circular field
3. Adjust the domain and grid size
4. Click "Plot Vector Field"

### Example: Solving a Differential Equation

1. Select the "Differential Equations" module
2. Enter an ODE in the form dy/dx = f(x,y)
   - For example: `y` for the equation dy/dx = y
3. Set the initial value, starting point, and end point
4. Select a solution method (Euler, Runge-Kutta, or Adaptive)
5. Click "Solve ODE"

## System Requirements

- Blender 3.0.0 or newer
- Python 3.7 or newer (included with Blender)
- NumPy (included with Blender)

## Project Structure

```markdown
math_playground/
  ├── __init__.py
  ├── properties.py
  ├── utils/
  │   ├── __init__.py
  │   ├── materials.py
  │   ├── collections.py
  │   ├── progress.py
  │   └── math_utils.py
  ├── operators/
  │   ├── __init__.py
  │   ├── linear_algebra.py
  │   ├── number_theory.py
  │   ├── analysis.py
  │   ├── graph_theory.py
  │   └── common.py
  ├── ui/
  │   ├── __init__.py
  │   ├── panels.py
  │   └── module_selectors.py
  └── algorithms/
      ├── __init__.py
      ├── differential.py
      ├── fourier.py
      ├── complex.py
      └── graph_algorithms.py
```

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- The Blender Foundation for creating an amazing 3D software
- The mathematical community for inspiration and algorithms
- All contributors to this project

## Future Development

Planned features for future releases:

- Topology visualization tools
- Quantum mechanics visualizations
- Statistical analysis and visualization
- Fractal geometry exploration
- Machine learning visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.
