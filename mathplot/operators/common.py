# operators/common.py - Common operators

import bpy
from bpy.types import Operator
from ..utils import collections

class MATH_OT_ClearAll(Operator):
    """Clear all math objects from the scene"""
    bl_idname = "math.clear_all"
    bl_label = "Clear All Math Objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Clear all math module collections
        collections.clear_module_collections()
        
        self.report({'INFO'}, "All math objects cleared")
        return {'FINISHED'}

# Registration functions
classes = [
    MATH_OT_ClearAll,
]

def register():
    """Register common operators"""
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    """Unregister common operators"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)