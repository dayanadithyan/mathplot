# utils/import_utils.py - Import utilities for Math Playground

import bpy
import importlib
import sys
import inspect
import os
from pathlib import Path

# ----------------------------------------
# Module Organization
# ----------------------------------------

ADDON_MODULES = {
    'utils': [
        'materials',
        'collections',
        'progress',
        'math_utils',
        'error_utils',
        'performance',
        'import_utils'
    ],
    'operators': [
        'linear_algebra',
        'number_theory',
        'analysis',
        'graph_theory',
        'common'
    ],
    'ui': [
        'panels',
        'module_selectors'
    ],
    'algorithms': [
        'differential',
        'fourier',
        'complex',
        'graph_algorithms'
    ]
}

# ----------------------------------------
# Import Functions
# ----------------------------------------

def get_addon_modules():
    """Get a list of all addon modules.
    
    Returns:
        list: List of module names
    """
    modules = []
    for category, module_list in ADDON_MODULES.items():
        for module in module_list:
            modules.append(f"mathplot.{category}.{module}")
    return modules

def reload_addon_modules():
    """Reload all addon modules, useful during development."""
    modules = get_addon_modules()
    
    # Add parent modules
    modules.append("mathplot.utils")
    modules.append("mathplot.operators")
    modules.append("mathplot.ui")
    modules.append("mathplot.algorithms")
    modules.append("mathplot")
    
    # Reload in reverse order to ensure dependencies are reloaded properly
    for module_name in reversed(modules):
        if module_name in sys.modules:
            try:
                importlib.reload(sys.modules[module_name])
                print(f"Reloaded module: {module_name}")
            except Exception as e:
                print(f"Error reloading module {module_name}: {e}")

def import_module(module_name, reload=False):
    """Import a module with explicit error handling.
    
    Args:
        module_name (str): Name of the module to import
        reload (bool): Whether to reload the module if already imported
        
    Returns:
        module: Imported module or None if import failed
    """
    try:
        if module_name in sys.modules and reload:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = importlib.import_module(module_name)
        return module
    except ImportError as e:
        print(f"Error importing module {module_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error importing {module_name}: {e}")
        return None

def import_from_module(module_name, names, reload=False):
    """Import specific names from a module.
    
    Args:
        module_name (str): Name of the module to import from
        names (list): List of names to import
        reload (bool): Whether to reload the module if already imported
        
    Returns:
        dict: Dictionary mapping names to imported objects
    """
    module = import_module(module_name, reload)
    if not module:
        return {}
    
    result = {}
    for name in names:
        if hasattr(module, name):
            result[name] = getattr(module, name)
        else:
            print(f"Warning: {name} not found in module {module_name}")
    
    return result

def get_module_classes(module_name, base_class=None):
    """Get all classes from a module, optionally filtering by base class.
    
    Args:
        module_name (str): Name of the module
        base_class (type, optional): Base class to filter by
        
    Returns:
        list: List of class objects
    """
    module = import_module(module_name)
    if not module:
        return []
    
    classes = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        # Skip imported classes (only include those defined in the module)
        if obj.__module__ != module_name:
            continue
            
        # Filter by base class if specified
        if base_class and not issubclass(obj, base_class):
            continue
            
        classes.append(obj)
    
    return classes

def get_addon_path():
    """Get the path to the addon directory.
    
    Returns:
        str: Path to the addon directory
    """
    # Get path to this file
    file_path = os.path.realpath(__file__)
    
    # Navigate up to the addon directory
    addon_path = os.path.dirname(os.path.dirname(file_path))
    
    return addon_path

def get_module_path(module_name):
    """Get the path to a module.
    
    Args:
        module_name (str): Name of the module
        
    Returns:
        str: Path to the module
    """
    try:
        module = import_module(module_name)
        if module and hasattr(module, "__file__"):
            return os.path.dirname(os.path.realpath(module.__file__))
        else:
            return None
    except:
        return None

def ensure_addon_in_path():
    """Ensure the addon directory is in the Python path."""
    addon_path = get_addon_path()
    if addon_path not in sys.path:
        sys.path.append(addon_path)
        print(f"Added {addon_path} to Python path")

# ----------------------------------------
# Safe Dynamic Import System
# ----------------------------------------

class AddonModuleManager:
    """Class to manage addon modules with proper dependency resolution."""
    
    def __init__(self):
        """Initialize the addon module manager."""
        self.imported_modules = {}
        self.module_status = {}  # 'pending', 'loading', 'loaded', 'error'
        self.module_errors = {}
    
    def import_module(self, module_name, reload=False):
        """Import a module with dependency resolution.
        
        Args:
            module_name (str): Name of the module to import
            reload (bool): Whether to reload the module and its dependencies
            
        Returns:
            module: Imported module or None if import failed
        """
        # Check if module already imported
        if module_name in self.imported_modules and not reload:
            return self.imported_modules[module_name]
        
        # Check for circular dependencies
        if self.module_status.get(module_name) == 'loading':
            print(f"Circular dependency detected for {module_name}")
            return None
        
        # Mark module as loading
        self.module_status[module_name] = 'loading'
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Reload if requested
            if reload and module_name in sys.modules:
                module = importlib.reload(module)
            
            # Store the imported module
            self.imported_modules[module_name] = module
            self.module_status[module_name] = 'loaded'
            
            return module
            
        except Exception as e:
            self.module_status[module_name] = 'error'
            self.module_errors[module_name] = str(e)
            print(f"Error importing module {module_name}: {e}")
            return None
    
    def import_all_modules(self, reload=False):
        """Import all addon modules.
        
        Args:
            reload (bool): Whether to reload all modules
            
        Returns:
            dict: Dictionary mapping module names to imported modules
        """
        modules = get_addon_modules()
        
        for module_name in modules:
            self.import_module(module_name, reload)
        
        return self.imported_modules
    
    def get_module_classes(self, module_name, base_class=None):
        """Get all classes from a module, optionally filtering by base class.
        
        Args:
            module_name (str): Name of the module
            base_class (type, optional): Base class to filter by
            
        Returns:
            list: List of class objects
        """
        module = self.import_module(module_name)
        if not module:
            return []
        
        classes = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes (only include those defined in the module)
            if obj.__module__ != module_name:
                continue
                
            # Filter by base class if specified
            if base_class and not issubclass(obj, base_class):
                continue
                
            classes.append(obj)
        
        return classes
    
    def reload_all_modules(self):
        """Reload all imported modules."""
        # Get current modules in reverse dependency order
        modules = list(self.imported_modules.keys())
        modules.reverse()
        
        # Clear the imported modules dictionary
        self.imported_modules = {}
        self.module_status = {}
        self.module_errors = {}
        
        # Import all modules again
        for module_name in modules:
            self.import_module(module_name, reload=True)
    
    def get_module_status_report(self):
        """Get a report of module import status.
        
        Returns:
            str: Status report
        """
        report = []
        for module_name, status in self.module_status.items():
            if status == 'error':
                error = self.module_errors.get(module_name, 'Unknown error')
                report.append(f"{module_name}: {status} - {error}")
            else:
                report.append(f"{module_name}: {status}")
        
        return "\n".join(report)

# Singleton instance
module_manager = AddonModuleManager()

# ----------------------------------------
# Optimized Plugin Interface
# ----------------------------------------

class PluginInterface:
    """Interface for loading and managing plugins."""
    
    def __init__(self, plugin_directory="plugins"):
        """Initialize the plugin interface.
        
        Args:
            plugin_directory (str): Directory to scan for plugins
        """
        self.plugin_directory = plugin_directory
        self.plugins = {}  # {name: module}
        self.plugin_info = {}  # {name: info_dict}
    
    def scan_plugins(self):
        """Scan for available plugins.
        
        Returns:
            list: List of plugin info dictionaries
        """
        # Get addon path
        addon_path = get_addon_path()
        plugin_path = os.path.join(addon_path, self.plugin_directory)
        
        # Ensure plugin directory exists
        if not os.path.exists(plugin_path):
            os.makedirs(plugin_path)
        
        # List to collect plugin info
        plugin_info_list = []
        
        # Check all python files in the plugin directory
        for file_path in Path(plugin_path).glob("*.py"):
            # Skip files starting with underscore
            if file_path.name.startswith("_"):
                continue
            
            # Get module name
            module_name = file_path.stem
            
            try:
                # Create a spec for the module
                spec = importlib.util.spec_from_file_location(
                    f"mathplot.{self.plugin_directory}.{module_name}", 
                    file_path
                )
                
                if spec:
                    # Import the module
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check if module has required attributes
                    if hasattr(module, "bl_info"):
                        # Get plugin info
                        plugin_info = module.bl_info.copy()
                        plugin_info["module_name"] = module_name
                        plugin_info["file_path"] = str(file_path)
                        
                        # Add to results
                        plugin_info_list.append(plugin_info)
                        self.plugin_info[module_name] = plugin_info
                        self.plugins[module_name] = module
            
            except Exception as e:
                print(f"Error scanning plugin {module_name}: {e}")
        
        return plugin_info_list
    
    def load_plugin(self, plugin_name):
        """Load a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin to load
            
        Returns:
            module: Loaded plugin module or None if loading failed
        """
        if plugin_name in self.plugins:
            return self.plugins[plugin_name]
        
        # Get plugin info
        plugin_info = self.plugin_info.get(plugin_name)
        if not plugin_info:
            print(f"Plugin {plugin_name} not found")
            return None
        
        # Import the plugin
        try:
            # Create a spec for the module
            spec = importlib.util.spec_from_file_location(
                f"mathplot.{self.plugin_directory}.{plugin_name}", 
                plugin_info["file_path"]
            )
            
            if spec:
                # Import the module
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Store the plugin
                self.plugins[plugin_name] = module
                
                return module
                
        except Exception as e:
            print(f"Error loading plugin {plugin_name}: {e}")
            return None
    
    def unload_plugin(self, plugin_name):
        """Unload a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin to unload
            
        Returns:
            bool: True if plugin was unloaded, False otherwise
        """
        if plugin_name not in self.plugins:
            return False
        
        # Get the module
        module = self.plugins[plugin_name]
        
        # Call unregister if available
        if hasattr(module, "unregister"):
            try:
                module.unregister()
            except Exception as e:
                print(f"Error unregistering plugin {plugin_name}: {e}")
        
        # Remove from dictionaries
        del self.plugins[plugin_name]
        
        # Remove from sys.modules if present
        module_name = f"mathplot.{self.plugin_directory}.{plugin_name}"
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        return True
    
    def reload_plugin(self, plugin_name):
        """Reload a specific plugin.
        
        Args:
            plugin_name (str): Name of the plugin to reload
            
        Returns:
            module: Reloaded plugin module or None if reloading failed
        """
        # Unload the plugin first
        if not self.unload_plugin(plugin_name):
            return None
        
        # Load the plugin again
        return self.load_plugin(plugin_name)
    
    def register_plugins(self, plugins=None):
        """Register all or specified plugins.
        
        Args:
            plugins (list, optional): List of plugin names to register
            
        Returns:
            int: Number of plugins registered
        """
        if plugins is None:
            # Register all plugins
            plugins = list(self.plugin_info.keys())
        
        registered_count = 0
        
        for plugin_name in plugins:
            # Load the plugin if not loaded
            if plugin_name not in self.plugins:
                module = self.load_plugin(plugin_name)
            else:
                module = self.plugins[plugin_name]
            
            if not module:
                continue
            
            # Register the plugin
            if hasattr(module, "register"):
                try:
                    module.register()
                    registered_count += 1
                except Exception as e:
                    print(f"Error registering plugin {plugin_name}: {e}")
        
        return registered_count
    
    def unregister_plugins(self, plugins=None):
        """Unregister all or specified plugins.
        
        Args:
            plugins (list, optional): List of plugin names to unregister
            
        Returns:
            int: Number of plugins unregistered
        """
        if plugins is None:
            # Unregister all plugins
            plugins = list(self.plugins.keys())
        
        unregistered_count = 0
        
        for plugin_name in plugins:
            if plugin_name not in self.plugins:
                continue
            
            # Get the module
            module = self.plugins[plugin_name]
            
            # Unregister the plugin
            if hasattr(module, "unregister"):
                try:
                    module.unregister()
                    unregistered_count += 1
                except Exception as e:
                    print(f"Error unregistering plugin {plugin_name}: {e}")
        
        return unregistered_count

# Singleton instance
plugin_interface = PluginInterface()

# ----------------------------------------
# Registration
# ----------------------------------------

def register():
    """Register import utilities"""
    # Ensure addon is in path
    ensure_addon_in_path()
    
    # Scan for plugins
    plugin_interface.scan_plugins()
    
    print("Math Playground: Import utilities registered")

def unregister():
    """Unregister import utilities"""
    # Unregister all plugins
    plugin_interface.unregister_plugins()
    
    print("Math Playground: Import utilities unregistered")