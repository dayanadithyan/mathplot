# Math Playground

Blender add-on for mathematical visualization in 3D space. Implements algorithms for linear algebra, number theory, analysis, graph theory, differential equations, complex analysis, and Fourier series.

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
mathplot/
├── __init__.py          # Registration and metadata
├── properties.py        # Property group definitions
├── algorithms/          # Mathematical implementations
├── operators/           # Blender operator implementations
├── ui/                  # Interface components
└── utils/               # Helper functions
```

## Dependencies

- Blender 3.0+
- NumPy (included with Blender)

## License

GPL-3.0

## Contributing

PRs welcome. Run tests with `pytest` before submission.
