# utils/__init__.py - Utility module initialization

from . import materials
from . import collections
from . import progress
from . import math_utils
from .collections import CollectionHelpers

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
