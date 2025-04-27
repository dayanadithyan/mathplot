# ui/__init__.py - UI module initialization

from mathplot.mathplot.ui import panels
from mathplot.mathplot.ui import module_selectors

def register():
    """Register all UI modules"""
    module_selectors.register()
    panels.register()

def unregister():
    """Unregister all UI modules"""
    panels.unregister()
    module_selectors.unregister()