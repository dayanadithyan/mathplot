#!/usr/bin/env python3

import os
import re
import ast
import shutil
import tempfile
from datetime import datetime
from typing import List, Dict, Set, Optional, Tuple, Any


def create_backup(base_dir: str) -> str:
    """Create a backup of the entire codebase."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"{base_dir}_backup_{timestamp}"

    # Create the backup directory
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Copy all files to the backup directory
    for root, dirs, files in os.walk(base_dir):
        # Skip the backup directory itself
        if root.startswith(backup_dir):
            continue

        # Create corresponding directories in backup
        rel_path = os.path.relpath(root, base_dir)
        backup_path = os.path.join(backup_dir, rel_path)
        if rel_path != "." and not os.path.exists(backup_path):
            os.makedirs(backup_path)

        # Copy files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(
    backup_path,
    file) if rel_path != "." else os.path.join(
        backup_dir,
         file)
            shutil.copy2(src_file, dst_file)

    print(f"Created backup at: {backup_dir}")
    return backup_dir


def get_python_files(directory: str) -> List[str]:
    """Get all Python files in the directory recursively."""
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def fix_imports(
    file_path: str, base_module_name: str = "mathplot") -> Tuple[bool, str]:
    """Fix incorrect import paths in a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Fix redundant module prefix like "mathplot.mathplot."
        redundant_prefix = f"{base_module_name}.{base_module_name}"
        pattern = re.compile(rf"from\s+{redundant_prefix}\.(\S+)\s+import")
        content = pattern.sub(f"from {base_module_name}.\\1 import", content)

        # Also fix imports in form "import mathplot.something"
        pattern2 = re.compile(rf"import\s+{redundant_prefix}\.(\S+)")
        content = pattern2.sub(f"import {base_module_name}.\\1", content)

        # Fix relative imports when appropriate
        # We need to determine the module path from the file path
        rel_path = os.path.relpath(
    file_path, os.path.dirname(
        os.path.dirname(file_path)))
        package_parts = os.path.dirname(rel_path).split(os.sep)

        if package_parts and package_parts[0] == base_module_name:
            # If in a subpackage, convert absolute imports within the same
            # package to relative
            for subpackage in package_parts[1:]:
                pattern = re.compile(
                    rf"from\s+{base_module_name}\.{subpackage}\s+import")
                content = pattern.sub(f"from . import", content)

        # Write the updated content
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fixed imports in {file_path}"

        return False, f"No import fixes needed in {file_path}"

    except Exception as e:
        return False, f"Error fixing imports in {file_path}: {str(e)}"


def add_missing_type_imports(file_path: str) -> Tuple[bool, str]:
    """Add missing type imports where needed."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Look for type annotations without imports
        type_imports_needed = set()

        # Check for various common type annotations
        type_patterns = {
            r'List\[': 'List',
            r'Dict\[': 'Dict',
            r'Optional\[': 'Optional',
            r'Tuple\[': 'Tuple',
            r'Set\[': 'Set',
            r'Callable\[': 'Callable',
            r'Any,': 'Any',
            r'Union\[': 'Union',
            r'TypeVar\(': 'TypeVar',
        }

        for pattern, type_name in type_patterns.items():
            if re.search(
    pattern,
     content) and f"from typing import {type_name}" not in content and f"from typing import " not in content:
                type_imports_needed.add(type_name)

        # Add the imports if needed
        if type_imports_needed:
            imports_line = f"from typing import {', '.join(sorted(type_imports_needed))}\n"

            # Try to find where to add the imports
            import_section_match = re.search(
    r'^import.*?\n\n', content, re.MULTILINE | re.DOTALL)
            if import_section_match:
                insertion_point = import_section_match.end()
                content = content[:insertion_point] + \
                    imports_line + content[insertion_point:]
            else:
                # Add after any module docstring
                docstring_match = re.match(
    r'^(""".*?"""\n)', content, re.DOTALL)
                if docstring_match:
                    insertion_point = docstring_match.end()
                    content = content[:insertion_point] + "\n" + \
                        imports_line + content[insertion_point:]
                else:
                    # Add at the beginning
                    content = imports_line + content

        # Write the updated content
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Added type imports in {file_path}: {', '.join(type_imports_needed)}"

        return False, f"No type imports needed in {file_path}"

    except Exception as e:
        return False, f"Error adding type imports in {file_path}: {str(e)}"


def fix_undefined_functions(file_path: str) -> Tuple[bool, str]:
    """Identify undefined functions referenced in the codebase and implement them."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # This is challenging to do automatically. We'll focus on common patterns.
        # For this example, we'll check for the validate_expression function in
        # error_utils.py
        if "error_utils.py" in file_path and "def validate_expression" not in content and "validate_expression" in content:
            # Implementation for validate_expression
            validate_expression_code = '''
def validate_expression(expression: str,
    required_vars: List[str],
    test_vars: Dict[str,
    Any]) -> Tuple[bool,
     str]:
    """Validate a mathematical expression.

    Args:
        expression: The expression to validate
        required_vars: List of variable names that should be used in the expression
        test_vars: Dictionary of test values for variables

    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Check if expression is empty
    if not expression or not expression.strip():
        return False, "Expression cannot be empty"

    # Check for required variables
    for var in required_vars:
        if var not in expression:
            return False, f"Expression should use the variable '{var}'"

    # Try to evaluate the expression with test values
    try:
        # Create the safe namespace with math functions
        import math
        import numpy as np

        safe_namespace = {
            "math": math,
            "np": np,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "pi": math.pi,
            "e": math.e,
            "abs": abs,
            "pow": pow,
            "min": min,
            "max": max,
            **test_vars
        }

        # Check for unsafe operations
        unsafe_terms = ["__", "import ", "eval(", "exec(", "compile(", "globals(", "locals(",
                       "getattr(", "setattr(", "delattr(", "open(", "file(", "os.", "sys."]
        for term in unsafe_terms:
            if term in expression:
                return False, f"Unsafe term detected in expression: {term}"

        # Try to evaluate
        eval(expression, {"__builtins__": {}}, safe_namespace)
        return True, ""

    except Exception as e:
        return False, f"Error evaluating expression: {str(e)}"
'''

            # Add the function implementation
            # Find a good place to add it - after imports but before other
            # functions
            match = re.search(r'import.*?\n\n', content, re.DOTALL)
            if match:
                insert_point = match.end()
                content = content[:insert_point] + \
                    validate_expression_code + "\n" + content[insert_point:]
            else:
                # Add at the end of imports
                lines = content.split("\n")
                import_lines = []
                non_import_lines = []

                for line in lines:
                    if line.strip().startswith(("import ", "from ")):
                        import_lines.append(line)
                    else:
                        non_import_lines.append(line)

                content = "\n".join(
                    import_lines) + "\n\n" + validate_expression_code + "\n\n" + "\n".join(non_import_lines)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True, f"Added validate_expression function to {file_path}"

        # Check for performance.py missing mesh functions
        if "performance.py" in file_path:
            missing_functions = []
            for func_name in ["create_cylinder_mesh", "create_cone_mesh", "create_uv_sphere_mesh",
                              "batch_create_objects", "instancing_create_objects", "create_lod_mesh"]:
                if func_name in content and f"def {func_name}" not in content:
                    missing_functions.append(func_name)

            if missing_functions:
                # We would implement these functions, but for brevity we'll
                # just report them
                return False, f"Missing function implementations in {file_path}: {', '.join(missing_functions)}"

        return False, f"No undefined functions fixed in {file_path}"

    except Exception as e:
        return False, f"Error fixing undefined functions in {file_path}: {str(e)}"


def improve_error_handling(file_path: str) -> Tuple[bool, str]:
    """Improve error handling in registration process."""
    try:
        if "__init__.py" in file_path and "def register" in file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Look for simple registration function without proper error
            # handling
            register_pattern = re.compile(
    r'def register\(\):(.*?)def unregister\(\):', re.DOTALL)
            register_match = register_pattern.search(content)

            if register_match:
                register_body = register_match.group(1)

                # Check if it already has comprehensive error handling
                if "try:" not in register_body or "except Exception as e:" not in register_body:
                    # Improved registration function with error handling
                    improved_register = '''def register():
    """Register the add-on with comprehensive error handling."""
    try:
        # Set up addon modules path
        setup_addon_modules()

        # Register properties first since they're used by operators and UI
        try:
            properties.register()
        except Exception as e:
            print(f"Error registering properties: {e}")
            raise

        # Register utility functions
        try:
            utils.register()
        except Exception as e:
            print(f"Error registering utils: {e}")
            raise

        # Register algorithms
        try:
            algorithms.register()
        except Exception as e:
            print(f"Error registering algorithms: {e}")
            raise

        # Register operators
        try:
            operators.register()
        except Exception as e:
            print(f"Error registering operators: {e}")
            raise

        # Register UI components last
        try:
            ui.register()
        except Exception as e:
            print(f"Error registering UI: {e}")
            raise

        # Create the main property group for scene
        bpy.types.Scene.math_playground = bpy.props.PointerProperty(
            type=properties.MathPlaygroundPropertyGroup)

        print(f"Math Playground {__version__} registered successfully")
        return True

    except Exception as e:
        # Get detailed error information
        import traceback
        error_msg = traceback.format_exc()

        # Print error details
        print(f"Error registering Math Playground: {e}")
        print(error_msg)

        # Show error in Blender UI if possible
        if hasattr(bpy.context, 'window_manager'):
            def draw_error(self, context):
                    """draw_error function.
    """
    self.layout.label(text=f"Error registering Math Playground: {e}")

            bpy.context.window_manager.popup_menu(
    draw_error, title="Registration Error", icon='ERROR')

        return False'''

                    # Replace the old register function with the improved one
                    content = register_pattern.sub(improved_register + "\n\ndef unregister():    """unregister function.
    """
    ", content)

                    # Write the updated content
                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        return True, f"Improved error handling in registration function in {file_path}"

            return False, f"No improvement needed for error handling in {file_path}"

        return False, f"Not a registration file: {file_path}"

    except Exception as e:
        return False, f"Error improving error handling in {file_path}: {str(e)}"

def fix_syntax_issues(file_path: str) -> Tuple[bool, str]:
    """Fix syntax issues and inconsistencies."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # Replace non-standard space characters
        # Replace non-breaking space with regular space
        content = content.replace('\u00a0', ' ')

        # Ensure consistent line endings
        content = content.replace('\r\n', '\n')

        # Fix missing or incorrect docstrings
        # This is a simple approach - ideally we would parse the AST and fix
        # more systematically
        function_def_pattern = re.compile(
    r'def\\s+(\\w+)\\s*\\(.*?\\):\\s*(?:""".*?"""\\s*)?', re.DOTALL)

        for match in function_def_pattern.finditer(content):
            func_name = match.group(1)
            func_text = match.group(0)

            # If function doesn't have a docstring, add a simple one
            if '"""' not in func_text:
                replacement=func_text +
                    f'    """{func_name} function.\n    """\n    '
                content=content.replace(func_text, replacement)

        # Write the updated content
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, f"Fixed syntax issues in {file_path}"

        return False, f"No syntax issues found in {file_path}"

    except Exception as e:
        return False, f"Error fixing syntax issues in {file_path}: {str(e)}"

def main():
    """Main function to execute the script."""
    # Get the directory to process
    script_dir=os.path.dirname(os.path.abspath(__file__))
    # Assuming the script is in a subdirectory
    base_dir=os.path.dirname(script_dir)

    # Ask for confirmation
    print(f"This script will modify Python files in: {base_dir}")
    confirm=input("Proceed? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    # Create backup
    backup_dir=create_backup(base_dir)

    # Get all Python files
    python_files=get_python_files(base_dir)
    print(f"Found {len(python_files)} Python files to process.")

    # Statistics for reporting
    stats={
        "imports_fixed": 0,
        "type_imports_added": 0,
        "undefined_functions_fixed": 0,
        "error_handling_improved": 0,
        "syntax_issues_fixed": 0,
        "files_processed": 0,
        "errors": []
    }

    # Process each file
    for file_path in python_files:
        print(f"\nProcessing {file_path}...")

        try:
            # Fix imports
            success, message=fix_imports(file_path)
            if success:
                stats["imports_fixed"] += 1
            print(message)

            # Add missing type imports
            success, message=add_missing_type_imports(file_path)
            if success:
                stats["type_imports_added"] += 1
            print(message)

            # Fix undefined functions
            success, message=fix_undefined_functions(file_path)
            if success:
                stats["undefined_functions_fixed"] += 1
            print(message)

            # Improve error handling
            success, message=improve_error_handling(file_path)
            if success:
                stats["error_handling_improved"] += 1
            print(message)

            # Fix syntax issues
            success, message=fix_syntax_issues(file_path)
            if success:
                stats["syntax_issues_fixed"] += 1
            print(message)

            stats["files_processed"] += 1

        except Exception as e:
            error_message=f"Error processing {file_path}: {str(e)}"
            print(error_message)
            stats["errors"].append(error_message)

    # Print summary
    print("\n" + "=" * 50)
    print("Processing complete. Summary:")
    print(
        f"Files processed: {stats['files_processed']} of {len(python_files)}")
    print(f"Import paths fixed: {stats['imports_fixed']}")
    print(f"Type imports added: {stats['type_imports_added']}")
    print(f"Undefined functions fixed: {stats['undefined_functions_fixed']}")
    print(f"Error handling improved: {stats['error_handling_improved']}")
    print(f"Syntax issues fixed: {stats['syntax_issues_fixed']}")

    if stats["errors"]:
        print(f"\nErrors encountered ({len(stats['errors'])}):")
        for error in stats["errors"]:
            print(f"- {error}")

    print(f"\nBackup created at: {backup_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()
