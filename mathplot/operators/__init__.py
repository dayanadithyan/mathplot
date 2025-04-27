# operators/__init__.py - Operators module initialization

from mathplot.mathplot.operators import linear_algebra
from mathplot.mathplot.operators import number_theory
from mathplot.mathplot.operators import analysis
from mathplot.mathplot.operators import graph_theory
from mathplot.mathplot.operators import common

def register():
    """Register all operator modules"""
    linear_algebra.register()
    number_theory.register()
    analysis.register()
    graph_theory.register()
    common.register()

def unregister():
    """Unregister all operator modules"""
    common.unregister()
    graph_theory.unregister()
    analysis.unregister()
    number_theory.unregister()
    linear_algebra.unregister()