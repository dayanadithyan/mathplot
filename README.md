UPDATE (4/25): Need to lint and cleanup sxyntax issues, else functional

# Math Playground - Learn to enjoy the 'mess'

Blender add-on for mathematical visualization in 3D space. Implements algorithms for linear algebra, number theory, analysis, graph theory, differential equations, complex analysis, and Fourier series. More to come.

![image](https://github.com/user-attachments/assets/3c16c966-3de1-49da-a203-fa7d607a27be)

## Features

- **Linear Algebra**: Vectors, matrix transformations
- **Number Theory**: Prime visualization, integer sequences
- **Analysis**: Function plotting, vector fields, parametric curves
- **Graph Theory**: Graph generation, path algorithms, layout optimization
- **Differential Equations**: Numerical solvers (Euler, RK4, adaptive)
- **Complex Analysis**: Domain coloring, Riemann sphere
- **Fourier Series**: Series approximation, component visualization

## Installation

```md
git clone https://github.com/user/mathplot.git
cd mathplot
cp -r mathplot ~/.config/blender/[version]/scripts/addons/
```

Or via Blender GUI: Edit → Preferences → Add-ons → Install → select `mathplot.zip`

## Usage

Accessible via N-panel in 3D View. Select mathematical domain, configure parameters, execute operations.

Example (vector field):

```md
1. Select "Analysis" module
2. Set expressions: X: `-y`, Y: `x`, Z: `0`
3. Execute "Plot Vector Field"
```

## Architecture

```md
/mathplot/
  ├── __init__.py - Package initialization
  │
  ├── algorithms/ - Mathematical implementations
  │   ├── __init__.py
  │   ├── complex.py - Complex analysis algorithms
  │   ├── differential.py - Differential equation solvers
  │   ├── fourier.py - Fourier series visualization
  │   └── graph_algorithms.py - Graph theory algorithms
  │
  ├── operators/ - Blender operator implementations
  │   ├── __init__.py
  │   ├── analysis.py - Function plotting, vector fields
  │   ├── common.py - General operators
  │   ├── graph_theory.py - Graph creation and algorithms
  │   ├── linear_algebra.py - Vector/matrix operations
  │   └── number_theory.py - Sequence generation
  │
  ├── ui/ - Interface components
  │   ├── __init__.py
  │   ├── module_selectors.py - Module switching
  │   └── panels.py - UI panel definitions
  │
  └── utils/ - Helper functions
      ├── __init__.py
      ├── collections.py - Blender collection management
      ├── error_utils.py - Error handling utilities
      ├── import_utils.py - Dynamic module loading
      ├── materials.py - Material creation and management
      ├── math_utils.py - Mathematical utility functions
      ├── performance.py - Optimization utilities
      └── progress.py - Progress reporting for long operations
```

## Dependencies

- Blender 4.0+
- NumPy (included with Blender)

## License

GPL-3.0

## Contributing

PRs welcome. Run tests with `pytest` before submission.
