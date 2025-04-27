# mathplot/utils/import_utils.py

import importlib
import sys
from typing import List, Dict, Any, Optional

def import_from_module(module_path: str, names: List[str] = None) -> Dict[str, Any]:
    """Import specific names from a module dynamically.
    
    Args:
        module_path (str): The path to the module (e.g., 'mathplot.utils.materials')
        names (List[str], optional): List of names to import from the module.
            If None, imports all public names (names not starting with '_').
    
    Returns:
        Dict[str, Any]: Dictionary mapping names to their respective objects from the module
    
    Raises:
        ImportError: If the module cannot be imported or if any name is not found
    """
    try:
        # Import the module
        module = importlib.import_module(module_path)
        
        # Determine which names to import
        if names is None:
            # Import all public names (those not starting with '_')
            names = [name for name in dir(module) if not name.startswith('_')]
        
        # Create a dictionary of imported names
        imported = {}
        for name in names:
            if hasattr(module, name):
                imported[name] = getattr(module, name)
            else:
                raise ImportError(f"Cannot import name '{name}' from '{module_path}'")
        
        return imported
    
    except ImportError as e:
        # Re-raise with additional context
        raise ImportError(f"Error importing from '{module_path}': {str(e)}") from e

def import_lazy(module_path: str, names: List[str] = None) -> Dict[str, Any]:
    """Create lazy loading proxies for module imports.
    
    This allows deferring the actual import until the objects are used,
    which can improve startup performance.
    
    Args:
        module_path (str): The path to the module (e.g., 'mathplot.utils.materials')
        names (List[str], optional): List of names to import from the module.
            If None, will not create any proxies until explicitly requested.
    
    Returns:
        Dict[str, Any]: Dictionary mapping names to their respective lazy-loading proxies
    """
    class LazyProxy:
        def __init__(self, module_path: str, name: str):
            self.module_path = module_path
            self.name = name
            self._obj = None
        
        def __call__(self, *args, **kwargs):
            if self._obj is None:
                module = importlib.import_module(self.module_path)
                self._obj = getattr(module, self.name)
            return self._obj(*args, **kwargs)
    
    # Create proxies for the requested names
    proxies = {}
    if names:
        for name in names:
            proxies[name] = LazyProxy(module_path, name)
    
    return proxies

def ensure_module_imported(module_path: str) -> None:
    """Ensure a module is imported, without returning anything.
    
    This is useful for modules that have side effects on import.
    
    Args:
        module_path (str): The path to the module to import
    
    Raises:
        ImportError: If the module cannot be imported
    """
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Error importing '{module_path}': {str(e)}") from e