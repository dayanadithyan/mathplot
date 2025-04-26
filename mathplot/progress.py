# utils/progress.py - Progress reporting utilities for Math Playground

import bpy
import time
import threading
from . import materials

# ----------------------------------------
# Progress Reporting Functions
# ----------------------------------------

def report_progress(context, progress, message):
    """Report progress to the user during long operations.
    
    Args:
        context (bpy.types.Context): Current context
        progress (float): Progress value between 0.0 and 1.0
        message (str): Progress message to display
    """
    # Ensure progress is in valid range
    progress = max(0.0, min(1.0, progress))
    
    # Update the window manager progress bar
    context.window_manager.progress_update(int(progress * 100))
    
    # Set status message in the header
    if hasattr(context.area, "header_text_set"):
        context.area.header_text_set(message)
    
    # Force UI update
    context.window_manager.progress_update(int(progress * 100))

def start_progress(context, title="Processing..."):
    """Start progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
        title (str): Title for the progress operation
    """
    context.window_manager.progress_begin(0, 100)
    if hasattr(context.area, "header_text_set"):
        context.area.header_text_set(title)

def end_progress(context):
    """End progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
    """
    context.window_manager.progress_end()
    if hasattr(context.area, "header_text_set"):
        context.area.header_text_set(None)

# ----------------------------------------
# Advanced Progress Visualization
# ----------------------------------------

class ProgressVisualizer:
    """Class to visualize progress in 3D view."""
    
    def __init__(self, context, steps=100, title="Processing..."):
        """Initialize progress visualizer.
        
        Args:
            context (bpy.types.Context): Current context
            steps (int): Total number of steps
            title (str): Title for the progress operation
        """
        self.context = context
        self.steps = steps
        self.current_step = 0
        self.title = title
        self.start_time = time.time()
        self.indicator_obj = None
        
        # Start progress reporting
        start_progress(context, title)
        
        # Create progress indicator in 3D view
        self.create_progress_indicator()
    
    def create_progress_indicator(self):
        """Create a visual progress indicator in the 3D view."""
        # Create a plane object
        bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 5))
        self.indicator_obj = bpy.context.active_object
        self.indicator_obj.name = "ProgressIndicator"
        
        # Create a material
        material = materials.create_material("ProgressIndicator_Material", (0.2, 0.6, 1.0, 0.8))
        materials.apply_material(self.indicator_obj, material)
        
        # Make semi-transparent
        material.blend_method = 'BLEND'
        
        # Scale to zero width initially
        self.indicator_obj.scale.x = 0.01
        self.indicator_obj.scale.y = 0.1
    
    def update(self, step, message=None):
        """Update progress.
        
        Args:
            step (int): Current step
            message (str, optional): Progress message
        """
        # Update current step
        self.current_step = min(step, self.steps)
        
        # Calculate progress
        progress = self.current_step / self.steps
        
        # Update standard progress
        if message is None:
            elapsed = time.time() - self.start_time
            eta = (elapsed / max(progress, 0.001)) * (1 - progress)
            message = f"{self.title} {progress:.0%} (ETA: {eta:.1f}s)"
        
        report_progress(self.context, progress, message)
        
        # Update visual indicator
        if self.indicator_obj:
            self.indicator_obj.scale.x = progress
    
    def finish(self):
        """Finish progress reporting."""
        end_progress(self.context)
        
        # Remove visual indicator
        if self.indicator_obj:
            bpy.data.objects.remove(self.indicator_obj, do_unlink=True)
            self.indicator_obj = None

# ----------------------------------------
# Threaded Progress Tasks
# ----------------------------------------

class ProgressThread(threading.Thread):
    """Thread class for executing time-consuming operations with progress reporting."""
    
    def __init__(self, context, task_func, task_args=(), task_kwargs=None, title="Processing..."):
        """Initialize progress thread.
        
        Args:
            context (bpy.types.Context): Current context
            task_func (callable): Function to execute
            task_args (tuple): Arguments for task function
            task_kwargs (dict): Keyword arguments for task function
            title (str): Title for the progress operation
        """
        threading.Thread.__init__(self)
        self.context = context
        self.task_func = task_func
        self.task_args = task_args
        self.task_kwargs = task_kwargs or {}
        self.title = title
        self.result = None
        self.error = None
        self.progress = 0.0
        self.message = title
        self.visualizer = None
        self.cancelled = False
        self.daemon = True  # Thread will be killed when main program exits
    
    def run(self):
        """Run the thread."""
        try:
            # Create progress visualizer
            self.visualizer = ProgressVisualizer(self.context, title=self.title)
            
            # Add progress callback to kwargs
            self.task_kwargs['progress_callback'] = self.update_progress
            
            # Run the task
            self.result = self.task_func(*self.task_args, **self.task_kwargs)
            
        except Exception as e:
            self.error = e
        finally:
            # Clean up visualizer
            if self.visualizer:
                self.visualizer.finish()
    
    def update_progress(self, progress, message=None):
        """Update progress from the task function.
        
        Args:
            progress (float): Progress value between 0.0 and 1.0
            message (str, optional): Progress message
        """
        self.progress = progress
        if message:
            self.message = message
        
        # Update visualizer
        if self.visualizer:
            self.visualizer.update(int(progress * 100), message)
        
        # Check if cancelled
        return not self.cancelled
    
    def cancel(self):
        """Cancel the thread."""
        self.cancelled = True

def run_with_progress(context, task_func, task_args=(), task_kwargs=None, title="Processing..."):
    """Run a function with progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
        task_func (callable): Function to execute
        task_args (tuple): Arguments for task function
        task_kwargs (dict): Keyword arguments for task function
        title (str): Title for the progress operation
        
    Returns:
        Any: Result of the task function
        
    Raises:
        Exception: Any exception raised by the task function
    """
    # Create and start thread
    thread = ProgressThread(context, task_func, task_args, task_kwargs, title)
    thread.start()
    
    # Wait for thread to finish
    while thread.is_alive():
        # Check for Blender cancel
        if context.window_manager.is_interface_locked():
            thread.cancel()
            thread.join()
            raise Exception("Operation cancelled by user")
        
        # Give Blender some time to update UI
        time.sleep(0.1)
    
    # Check for errors
    if thread.error:
        raise thread.error
    
    return thread.result

# ----------------------------------------
# Modal Operator for Progress
# ----------------------------------------

class MATH_OT_ProgressModal(bpy.types.Operator):
    """Modal operator for progress reporting"""
    bl_idname = "math.progress_modal"
    bl_label = "Processing..."
    bl_options = {'REGISTER', 'INTERNAL'}
    
    progress: bpy.props.FloatProperty(
        name="Progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='PERCENTAGE'
    )
    
    message: bpy.props.StringProperty(
        name="Message",
        default="Processing..."
    )
    
    cancelled: bpy.props.BoolProperty(
        name="Cancelled",
        default=False
    )
    
    # Internal variables
    _timer = None
    _thread = None
    _task_func = None
    _task_args = None
    _task_kwargs = None
    _callback = None
    
    def modal(self, context, event):
        """Modal function for handling events.
        
        Args:
            context (bpy.types.Context): Current context
            event (bpy.types.Event): Event
            
        Returns:
            set: Operator return state
        """
        # Update progress bar
        if self._thread and self._thread.is_alive():
            self.progress = self._thread.progress
            self.message = self._thread.message
            context.area.header_text_set(self.message)
            
            # Handle cancellation
            if event.type == 'ESC':
                self.cancelled = True
                self._thread.cancel()
                self.cancel(context)
                return {'CANCELLED'}
        else:
            # Task completed or failed
            if self._thread and self._thread.error:
                self.report({'ERROR'}, str(self._thread.error))
                self.cancel(context)
                return {'CANCELLED'}
            elif self._thread:
                # Task completed successfully
                if self._callback:
                    try:
                        self._callback(self._thread.result)
                    except Exception as e:
                        self.report({'ERROR'}, str(e))
                
                self.cancel(context)
                return {'FINISHED'}
        
        return {'RUNNING_MODAL'}
    
    def execute(self, context):
        """Execute the operator.
        
        Args:
            context (bpy.types.Context): Current context
            
        Returns:
            set: Operator return state
        """
        # Start timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        
        # Start thread
        if self._task_func:
            self._thread = ProgressThread(
                context, 
                self._task_func, 
                self._task_args or (), 
                self._task_kwargs or {}, 
                self.message
            )
            self._thread.start()
        
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        """Cancel the operator.
        
        Args:
            context (bpy.types.Context): Current context
        """
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
        
        # Remove progress bar
        context.area.header_text_set(None)
        
        # Cancel thread if running
        if self._thread and self._thread.is_alive():
            self._thread.cancel()
            self._thread.join(1.0)  # Wait for thread to finish with timeout

def run_modal_with_progress(context, task_func, task_args=None, task_kwargs=None, title="Processing...", callback=None):
    """Run a function with modal progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
        task_func (callable): Function to execute
        task_args (tuple): Arguments for task function
        task_kwargs (dict): Keyword arguments for task function
        title (str): Title for the progress operation
        callback (callable): Function to call with the result
    """
    # Prepare operator
    MATH_OT_ProgressModal._task_func = task_func
    MATH_OT_ProgressModal._task_args = task_args
    MATH_OT_ProgressModal._task_kwargs = task_kwargs
    MATH_OT_ProgressModal._callback = callback
    
    # Run operator
    bpy.ops.math.progress_modal(message=title)

# ----------------------------------------
# Registration
# ----------------------------------------

classes = [
    MATH_OT_ProgressModal,
]

def register():
    """Register progress utilities"""
    for cls in classes:
        bpy.utils.register_class(cls)
    print("Math Playground: Progress utilities registered")

def unregister():
    """Unregister progress utilities"""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Math Playground: Progress utilities unregistered")