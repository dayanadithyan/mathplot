# operators/__init__.py - Operators module initialization

from . import linear_algebra
from . import number_theory
from . import analysis
from . import graph_theory
from . import common

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