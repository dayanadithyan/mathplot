# algorithms/__init__.py - Algorithms module initialization

from mathplot.mathplot.algorithms import differential
from mathplot.mathplot.algorithms import fourier
from mathplot.mathplot.algorithms import complex
from mathplot.mathplot.algorithms import graph_algorithms

def register():
    """Register all algorithm modules"""
    differential.register()
    fourier.register()
    complex.register()
    graph_algorithms.register()

def unregister():
    """Unregister all algorithm modules"""
    graph_algorithms.unregister()
    complex.unregister()
    fourier.unregister()
    differential.unregister()