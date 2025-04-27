# mathplot/utils/progress.py

import bpy
import time
import threading
from typing import Callable, Optional, List, Any, Dict, Union, Tuple
from mathutils import Vector
import traceback

# ----------------------------------------
# Thread-Safe Progress Reporting Functions
# ----------------------------------------


def report_progress(context: bpy.types.Context,
                    progress: float,
                    message: str,
                    allow_cancel: bool = True) -> bool:
    """Report progress to the user during long operations.

    Args:
        context: Current context
        progress: Progress value between 0.0 and 1.0
        message: Progress message to display
        allow_cancel: Whether to allow cancellation

    Returns:
        bool: False if the operation was cancelled, True otherwise
    """
    # Ensure progress is in valid range
    progress = max(0.0, min(1.0, progress))

    # Update the window manager progress bar
    context.window_manager.progress_update(int(progress * 100))

    # Set status message in the header
    if hasattr(context, "area") and context.area:
        if hasattr(context.area, "header_text_set"):
            context.area.header_text_set(message)

    # Force UI update
    context.window_manager.progress_update(int(progress * 100))

    # Check if operation was cancelled
    if allow_cancel and context.window_manager.is_interface_locked():
        return False

    return True


def start_progress(
        context: bpy.types.Context,
        title: str = "Processing...") -> None:
    """Start progress reporting.

    Args:
        context: Current context
        title: Title for the progress operation
    """
    context.window_manager.progress_begin(0, 100)
    if hasattr(context, "area") and context.area:
        if hasattr(context.area, "header_text_set"):
            context.area.header_text_set(title)


def end_progress(context: bpy.types.Context) -> None:
    """End progress reporting.

    Args:
        context: Current context
    """
    context.window_manager.progress_end()
    if hasattr(context, "area") and context.area:
        if hasattr(context.area, "header_text_set"):
            context.area.header_text_set(None)

# ----------------------------------------
# Batch Processing Functions
# ----------------------------------------


def batch_process(items: List[Any],
                  process_func: Callable[[Any], Any],
                  context: bpy.types.Context,
                  message: str = "Processing batch",
                  modal: bool = False,
                  chunk_size: int = 10) -> List[Any]:
    """Process a batch of items with progress reporting.

    Args:
        items: List of items to process
        process_func: Function to process each item
        context: Current context
        message: Progress message
        modal: Whether to use modal processing (yields to UI between chunks)
        chunk_size: Number of items to process in each chunk

    Returns:
        List of processed items
    """
    if not items:
        return []

    results = []
    total_items = len(items)
    processed = 0

    # Start progress
    start_progress(context, message)

    try:
        # Process in chunks
        for i in range(0, total_items, chunk_size):
            # Process a chunk
            chunk = items[i:i + chunk_size]
            for item in chunk:
                results.append(process_func(item))
                processed += 1

                # Report progress
                if not report_progress(
                        context,
                        processed / total_items,
                        f"{message}: {processed}/{total_items}"):
                    # User cancelled
                    end_progress(context)
                    return results

            # Yield to UI if modal
            if modal:
                # Update UI
                context.window_manager.progress_update(
                    int(processed / total_items * 100))
                # Give Blender time to update UI and process events
                if not process_modal_events(context):
                    # User cancelled
                    end_progress(context)
                    return results

    finally:
        # End progress
        end_progress(context)

    return results


def process_modal_events(context: bpy.types.Context) -> bool:
    """Process modal events to keep the UI responsive.

    Args:
        context: Current context

    Returns:
        bool: False if the operation was cancelled, True otherwise
    """
    # Give Blender time to update UI
    time.sleep(0.01)

    # Check if operation was cancelled
    if context.window_manager.is_interface_locked():
        return False

    return True

# ----------------------------------------
# Threaded Progress Tasks
# ----------------------------------------


class ProgressThread(threading.Thread):
    """Thread class for executing time-consuming operations with progress reporting."""

    def __init__(self, context: bpy.types.Context,
                 task_func: Callable[..., Any],
                 task_args: Tuple = (),
                 task_kwargs: Optional[Dict[str, Any]] = None,
                 title: str = "Processing..."):
        """Initialize progress thread.

        Args:
            context: Current context
            task_func: Function to execute
            task_args: Arguments for task function
            task_kwargs: Keyword arguments for task function
            title: Title for the progress operation
        """
        threading.Thread.__init__(self)
        self.context = context
        self.task_func = task_func
        self.task_args = task_args
        self.task_kwargs = task_kwargs or {}
        self.title = title
        self.result = None
        self.error = None
        self.exception_traceback = None
        self.progress = 0.0
        self.message = title
        self.cancelled = False
        self.daemon = True  # Thread will be killed when main program exits

    def run(self):
        """Run the thread."""
        try:
            # Add progress callback to kwargs
            self.task_kwargs['progress_callback'] = self.update_progress

            # Run the task
            self.result = self.task_func(*self.task_args, **self.task_kwargs)

        except Exception as e:
            self.error = e
            self.exception_traceback = traceback.format_exc()

    def update_progress(
            self,
            progress: float,
            message: Optional[str] = None) -> bool:
        """Update progress from the task function.

        Args:
            progress: Progress value between 0.0 and 1.0
            message: Progress message

        Returns:
            bool: False if the operation was cancelled, True otherwise
        """
        self.progress = progress
        if message:
            self.message = message

        # Check if cancelled
        return not self.cancelled

    def cancel(self):
        """Cancel the thread."""
        self.cancelled = True


def run_with_progress(context: bpy.types.Context,
                      task_func: Callable[..., Any],
                      task_args: Tuple = (),
                      task_kwargs: Optional[Dict[str, Any]] = None,
                      title: str = "Processing...") -> Any:
    """Run a function with progress reporting.

    Args:
        context: Current context
        task_func: Function to execute
        task_args: Arguments for task function
        task_kwargs: Keyword arguments for task function
        title: Title for the progress operation

    Returns:
        Result of the task function

    Raises:
        Exception: Any exception raised by the task function
    """
    # Start progress
    start_progress(context, title)

    try:
        # Create and start thread
        thread = ProgressThread(
            context,
            task_func,
            task_args,
            task_kwargs,
            title)
        thread.start()

        # Wait for thread to finish
        while thread.is_alive():
            # Report progress
            report_progress(context, thread.progress, thread.message)

            # Check for Blender cancel
            if context.window_manager.is_interface_locked():
                thread.cancel()
                thread.join()
                raise Exception("Operation cancelled by user")

            # Give Blender some time to update UI
            time.sleep(0.1)

        # Check for errors
        if thread.error:
            if thread.exception_traceback:
                print(thread.exception_traceback)
            raise thread.error

        return thread.result

    finally:
        # End progress
        end_progress(context)
