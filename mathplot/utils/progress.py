# utils/progress.py - Progress reporting utilities for Math Playground

import bpy
import time
import threading
import traceback
from threading import Lock
from mathutils import Vector

# ----------------------------------------
# Thread-safe Progress Reporting
# ----------------------------------------

# Global lock for progress updates
_progress_lock = Lock()

# Global state for progress reporting
_progress_state = {
    'active': False,
    'value': 0.0,
    'message': '',
    'canceled': False
}

def report_progress(context, progress, message):
    """Report progress to the user during long operations in a thread-safe manner.
    
    Args:
        context (bpy.types.Context): Current context
        progress (float): Progress value between 0.0 and 1.0
        message (str): Progress message to display
        
    Returns:
        bool: False if operation was canceled, True otherwise
    """
    # Ensure progress is in valid range
    progress = max(0.0, min(1.0, progress))
    
    # Only report if we have a valid context
    if context is None:
        return not _progress_state['canceled']
    
    # Update progress state in a thread-safe manner
    with _progress_lock:
        if not _progress_state['active']:
            return not _progress_state['canceled']
            
        _progress_state['value'] = progress
        _progress_state['message'] = message
        
        # Check if we're in the main thread
        is_main_thread = threading.current_thread() is threading.main_thread()
        
        if is_main_thread:
            # We can safely update UI from the main thread
            try:
                # Update the window manager progress bar
                context.window_manager.progress_update(int(progress * 100))
                
                # Set status message in the header
                if hasattr(context, "area") and context.area:
                    if hasattr(context.area, "header_text_set"):
                        context.area.header_text_set(message)
            except ReferenceError:
                # Context might have become invalid
                pass
    
    # Return cancel state
    return not _progress_state['canceled']

def start_progress(context, title="Processing..."):
    """Start progress reporting in a thread-safe manner.
    
    Args:
        context (bpy.types.Context): Current context
        title (str): Title for the progress operation
    """
    with _progress_lock:
        _progress_state['active'] = True
        _progress_state['value'] = 0.0
        _progress_state['message'] = title
        _progress_state['canceled'] = False
        
    # Only modify UI from the main thread
    if threading.current_thread() is threading.main_thread() and context:
        try:
            context.window_manager.progress_begin(0, 100)
            if hasattr(context, "area") and context.area:
                if hasattr(context.area, "header_text_set"):
                    context.area.header_text_set(title)
        except ReferenceError:
            # Context might have become invalid
            pass

def end_progress(context):
    """End progress reporting in a thread-safe manner.
    
    Args:
        context (bpy.types.Context): Current context
    """
    with _progress_lock:
        _progress_state['active'] = False
    
    # Only modify UI from the main thread
    if threading.current_thread() is threading.main_thread() and context:
        try:
            context.window_manager.progress_end()
            if hasattr(context, "area") and context.area:
                if hasattr(context.area, "header_text_set"):
                    context.area.header_text_set(None)
        except ReferenceError:
            # Context might have become invalid
            pass

def cancel_progress():
    """Cancel the current progress operation."""
    with _progress_lock:
        _progress_state['canceled'] = True

def is_progress_canceled():
    """Check if the current progress operation was canceled.
    
    Returns:
        bool: True if canceled, False otherwise
    """
    with _progress_lock:
        return _progress_state['canceled']

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
        self.steps = max(1, steps)
        self.current_step = 0
        self.title = title
        self.start_time = time.time()
        self.indicator_obj = None
        self._is_canceled = False
        
        # Start progress reporting
        start_progress(context, title)
        
        # Only create visual indicator from the main thread
        if threading.current_thread() is threading.main_thread():
            self._create_progress_indicator()
    
    def _create_progress_indicator(self):
        """Create a visual progress indicator in the 3D view."""
        try:
            # Create a plane object
            bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 5))
            self.indicator_obj = bpy.context.active_object
            self.indicator_obj.name = "ProgressIndicator"
            
            # Create a material
            material = bpy.data.materials.new("ProgressIndicator_Material")
            material.diffuse_color = (0.2, 0.6, 1.0, 0.8)
            
            # Apply material
            if self.indicator_obj.data.materials:
                self.indicator_obj.data.materials[0] = material
            else:
                self.indicator_obj.data.materials.append(material)
            
            # Make semi-transparent
            material.blend_method = 'BLEND'
            
            # Scale to zero width initially
            self.indicator_obj.scale.x = 0.01
            self.indicator_obj.scale.y = 0.1
        except Exception as e:
            print(f"Error creating progress indicator: {e}")
            self.indicator_obj = None
    
    def update(self, step, message=None):
        """Update progress.
        
        Args:
            step (int): Current step
            message (str, optional): Progress message
            
        Returns:
            bool: False if canceled, True otherwise
        """
        # Check if canceled
        if is_progress_canceled():
            self._is_canceled = True
            return False
            
        # Update current step
        self.current_step = min(step, self.steps)
        
        # Calculate progress
        progress = self.current_step / self.steps
        
        # Update standard progress
        if message is None:
            elapsed = time.time() - self.start_time
            eta = (elapsed / max(progress, 0.001)) * (1 - progress)
            message = f"{self.title} {progress:.0%} (ETA: {eta:.1f}s)"
        
        # Report progress (thread-safe)
        if not report_progress(self.context, progress, message):
            self._is_canceled = True
            return False
        
        # Update visual indicator only from the main thread
        if threading.current_thread() is threading.main_thread():
            if self.indicator_obj:
                try:
                    self.indicator_obj.scale.x = progress
                except ReferenceError:
                    # Object might have been deleted
                    self.indicator_obj = None
        
        return True
    
    def finish(self):
        """Finish progress reporting."""
        end_progress(self.context)
        
        # Remove visual indicator only from the main thread
        if threading.current_thread() is threading.main_thread():
            if self.indicator_obj:
                try:
                    bpy.data.objects.remove(self.indicator_obj, do_unlink=True)
                except ReferenceError:
                    # Object might have been deleted already
                    pass
                finally:
                    self.indicator_obj = None
    
    def is_canceled(self):
        """Check if the progress was canceled.
        
        Returns:
            bool: True if canceled, False otherwise
        """
        return self._is_canceled or is_progress_canceled()

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
        self.daemon = True  # Thread will be killed when main program exits
        self._is_finished = False
        self._lock = threading.RLock()  # Reentrant lock for thread safety
    
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
            self.error_traceback = traceback.format_exc()
            print(f"Error in progress thread: {e}\n{self.error_traceback}")
        finally:
            # Clean up visualizer
            if self.visualizer:
                self.visualizer.finish()
            
            with self._lock:
                self._is_finished = True
    
    def update_progress(self, progress, message=None):
        """Update progress from the task function.
        
        Args:
            progress (float): Progress value between 0.0 and 1.0
            message (str, optional): Progress message
            
        Returns:
            bool: False if canceled, True otherwise
        """
        with self._lock:
            self.progress = progress
            if message:
                self.message = message
            
            # Update visualizer
            if self.visualizer and not self.visualizer.update(int(progress * 100), message):
                return False
            
            # Check if canceled
            return not self.is_canceled()
    
    def cancel(self):
        """Cancel the thread."""
        cancel_progress()
    
    def is_canceled(self):
        """Check if the thread was canceled.
        
        Returns:
            bool: True if canceled, False otherwise
        """
        if self.visualizer:
            return self.visualizer.is_canceled()
        return is_progress_canceled()
    
    def is_finished(self):
        """Check if the thread has finished.
        
        Returns:
            bool: True if finished, False otherwise
        """
        with self._lock:
            return self._is_finished

def run_with_progress(context, task_func, task_args=(), task_kwargs=None, title="Processing...", timeout=None):
    """Run a function with progress reporting.
    
    Args:
        context (bpy.types.Context): Current context
        task_func (callable): Function to execute
        task_args (tuple): Arguments for task function
        task_kwargs (dict): Keyword arguments for task function
        title (str): Title for the progress operation
        timeout (float, optional): Timeout in seconds
        
    Returns:
        Any: Result of the task function
        
    Raises:
        Exception: Any exception raised by the task function
        TimeoutError: If the operation times out
    """
    # Create and start thread
    thread = ProgressThread(context, task_func, task_args, task_kwargs or {}, title)
    thread.start()
    
    start_time = time.time()
    
    # Wait for thread to finish
    while thread.is_alive():
        # Check for timeout
        if timeout and time.time() - start_time > timeout:
            thread.cancel()
            raise TimeoutError(f"Operation timed out after {timeout} seconds")
        
        # Check for Blender cancel
        if context and hasattr(context, 'window_manager') and context.window_manager.is_interface_locked():
            thread.cancel()
            thread.join(1.0)  # Give thread a second to clean up
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
    
    # Internal variables
    _timer = None
    _thread = None
    _task_func = None
    _task_args = None
    _task_kwargs = None
    _callback = None
    _is_cancelled = False
    
    def modal(self, context, event):
        """Modal function for handling events.
        
        Args:
            context (bpy.types.Context): Current context
            event (bpy.types.Event): Event
            
        Returns:
            set: Operator return state
        """
        # Check for escape key to cancel
        if event.type == 'ESC':
            self._is_cancelled = True
            if self._thread and self._thread.is_alive():
                self._thread.cancel()
            self.cancel(context)
            self.report({'INFO'}, "Operation cancelled by user")
            return {'CANCELLED'}
            
        # Update progress bar if thread is running
        if self._thread and self._thread.is_alive():
            with _progress_lock:
                self.progress = _progress_state['value']
                self.message = _progress_state['message']
                
                if _progress_state['canceled']:
                    self._is_cancelled = True
                    self.cancel(context)
                    self.report({'INFO'}, "Operation cancelled")
                    return {'CANCELLED'}
            
            # Update UI display
            if context.area:
                context.area.header_text_set(self.message)
            
            return {'RUNNING_MODAL'}
        else:
            # Thread has completed or failed
            if self._is_cancelled:
                self.cancel(context)
                return {'CANCELLED'}
                
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
            else:
                # No thread was running
                self.cancel(context)
                return {'CANCELLED'}
    
    def execute(self, context):
        """Execute the operator.
        
        Args:
            context (bpy.types.Context): Current context
            
        Returns:
            set: Operator return state
        """
        # Reset cancelled flag
        self._is_cancelled = False
        
        # Start timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        
        # Initialize progress
        start_progress(context, self.message)
        
        # Start thread if task function is provided
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
        
        # End progress
        end_progress(context)
        
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
# Utilities for batch processing with progress
# ----------------------------------------

def batch_process(items, process_func, context=None, title="Processing batch", 
                 modal=False, callback=None, chunk_size=10):
    """Process a batch of items with progress reporting.
    
    Args:
        items (list): List of items to process
        process_func (callable): Function to process each item
        context (bpy.types.Context, optional): Current context
        title (str): Title for the progress operation
        modal (bool): Whether to use modal progress reporting
        callback (callable, optional): Function to call when done (for modal processing)
        chunk_size (int): Number of items to process in each chunk
        
    Returns:
        list: Results of processing each item (if not modal)
    """
    def batch_task(progress_callback=None):
        results = []
        total = len(items)
        
        # Process in chunks to reduce UI update overhead
        for i in range(0, total, chunk_size):
            chunk = items[i:i+chunk_size]
            chunk_results = []
            
            for j, item in enumerate(chunk):
                # Process item
                result = process_func(item)
                chunk_results.append(result)
                
                # Report progress for each item
                current = i + j + 1
                if progress_callback:
                    if not progress_callback(current / total, f"Processing item {current}/{total}"):
                        return results  # Cancelled
            
            # Extend results after each chunk
            results.extend(chunk_results)
            
            # Force a redraw for every chunk
            if context and hasattr(context, 'area'):
                context.area.tag_redraw()
        
        return results
    
    if modal:
        # Use modal progress reporting
        run_modal_with_progress(context, batch_task, title=title, callback=callback)
        return None
    else:
        # Use non-modal progress reporting
        return run_with_progress(context, batch_task, title=title)

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