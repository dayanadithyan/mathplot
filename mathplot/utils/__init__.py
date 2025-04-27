# utils/__init__.py - Utility module initialization

from mathplot.utils import materials
from mathplot.utils import collections
from mathplot.utils import progress
from mathplot.utils import math_utils
from mathplot.utils.collections import CollectionHelpers

__all__ = ["CollectionHelpers"]


def register():
    """Register all utility modules"""
    materials.register()
    collections.register()
    progress.register()
    math_utils.register()


def unregister():
    """Unregister all utility modules"""
    math_utils.unregister()
    progress.unregister()
    collections.unregister()
    materials.unregister()
