"""
Subprocess utilities for ensuring child processes are properly cleaned up.

On Windows, uses Job Objects to automatically kill all child processes when the parent dies.
This works even if the parent crashes or is killed via Task Manager.
"""

import os
import sys
import logging
import subprocess
import atexit
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Windows Job Object support
_job_handle = None

if sys.platform == 'win32':
    import ctypes
    from ctypes import wintypes

    # Windows API constants
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001

    # Structures for Job Object configuration
    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_uint64),
            ('WriteOperationCount', ctypes.c_uint64),
            ('OtherOperationCount', ctypes.c_uint64),
            ('ReadTransferCount', ctypes.c_uint64),
            ('WriteTransferCount', ctypes.c_uint64),
            ('OtherTransferCount', ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', ctypes.c_int64),
            ('PerJobUserTimeLimit', ctypes.c_int64),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.POINTER(ctypes.c_ulong)),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32

    def create_job_object():
        """Create a Windows Job Object that kills children when closed."""
        global _job_handle

        if _job_handle is not None:
            return _job_handle

        # Create the job object
        _job_handle = kernel32.CreateJobObjectW(None, None)
        if not _job_handle:
            return None

        # Configure it to kill all processes when the job is closed
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        success = kernel32.SetInformationJobObject(
            _job_handle,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info)
        )

        if not success:
            kernel32.CloseHandle(_job_handle)
            _job_handle = None
            return None

        return _job_handle

    def assign_process_to_job(process):
        """
        Assign a process to the job object so it gets killed when the parent dies.

        Returns:
            True if assignment succeeded, False otherwise
        """
        job = create_job_object()
        if not job or not process:
            return False

        try:
            handle = int(process._handle)  # Get Windows process handle
            result = kernel32.AssignProcessToJobObject(job, handle)
            if not result:
                # AssignProcessToJobObject returns 0 on failure
                # This can happen if the process is already in another job
                # (common on Windows 8+ where processes are often in a default job)
                error_code = kernel32.GetLastError()
                # Error code 5 (ACCESS_DENIED) is common when process is in nested job
                # Error code 87 (ERROR_INVALID_PARAMETER) can occur with invalid handles
                logger.warning(f"Failed to assign process {process.pid} to job object "
                           f"(error code: {error_code}). Child processes may not be "
                           f"automatically cleaned up on parent exit.")
                return False
            return True
        except (AttributeError, ValueError) as e:
            # Process doesn't have a handle or it's invalid
            logger.warning(f"Could not get process handle for job assignment: {e}")
            return False

    def cleanup_job():
        """Clean up the job object (also kills all assigned processes)."""
        global _job_handle
        if _job_handle:
            kernel32.CloseHandle(_job_handle)
            _job_handle = None

    # Register cleanup on exit
    atexit.register(cleanup_job)

else:
    # Non-Windows platforms - use process groups instead of Job Objects
    def create_job_object():
        """No-op on non-Windows platforms."""
        return None

    def assign_process_to_job(process):
        """
        No-op on non-Windows platforms.
        Process groups are used instead via start_new_session=True.

        Returns:
            True (process groups are set up in create_subprocess_with_cleanup)
        """
        return True

    def cleanup_job():
        """No-op on non-Windows platforms."""
        pass


def create_subprocess_with_cleanup(args: List[str], **kwargs) -> subprocess.Popen:
    """
    Create a subprocess that will be automatically killed when the parent process dies.

    On Windows, this uses Job Objects for reliable cleanup even on crashes.
    On other platforms, uses process groups.

    Args:
        args: Command and arguments (same as subprocess.Popen)
        **kwargs: Additional arguments passed to subprocess.Popen

    Returns:
        subprocess.Popen object
    """
    if sys.platform == 'win32':
        # CREATE_NEW_PROCESS_GROUP (0x200) allows sending CTRL_BREAK_EVENT for graceful shutdown
        # CREATE_BREAKAWAY_FROM_JOB is NOT set so it stays in our job
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        creation_flags = kwargs.get('creationflags', 0)
        creation_flags |= CREATE_NEW_PROCESS_GROUP
        kwargs['creationflags'] = creation_flags

        process = subprocess.Popen(args, **kwargs)
        assign_process_to_job(process)
        return process
    else:
        # On Unix, use process groups for cleanup
        kwargs['start_new_session'] = True
        return subprocess.Popen(args, **kwargs)


def run_subprocess_with_cleanup(args: List[str], **kwargs) -> subprocess.CompletedProcess:
    """
    Run a subprocess (blocking) that will be killed if the parent dies.

    This is like subprocess.run() but ensures cleanup on parent death.

    Args:
        args: Command and arguments (same as subprocess.run)
        **kwargs: Additional arguments passed to subprocess.run

    Returns:
        subprocess.CompletedProcess object
    """
    if sys.platform == 'win32':
        # For subprocess.run, we need to manually manage the process
        timeout = kwargs.pop('timeout', None)

        process = create_subprocess_with_cleanup(
            args,
            stdout=kwargs.get('stdout'),
            stderr=kwargs.get('stderr'),
            stdin=kwargs.get('stdin'),
            env=kwargs.get('env'),
            cwd=kwargs.get('cwd'),
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        except (OSError, ValueError):
            # Ensure process is cleaned up on I/O or communication errors
            process.kill()
            process.wait()
            raise
    else:
        return subprocess.run(args, **kwargs)


def terminate_process_tree(process: Optional[subprocess.Popen]) -> None:
    """
    Terminate a process and all its children (forcefully).

    Args:
        process: subprocess.Popen object, or None (no-op).
    """
    if process is None:
        return

    if sys.platform == 'win32':
        try:
            # Use taskkill to kill the process tree
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            # Fallback to basic terminate
            logger.debug(f"taskkill failed for PID {process.pid}: {e}")
            try:
                process.terminate()
            except Exception as e2:
                logger.warning(f"terminate() also failed for PID {process.pid}: {e2}")
    else:
        # On Unix, kill the process group
        import signal
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass  # Process already exited - expected
        except Exception as e:
            logger.debug(f"killpg failed for PID {process.pid}: {e}")
            try:
                process.terminate()
            except Exception as e2:
                logger.warning(f"terminate() also failed for PID {process.pid}: {e2}")


def graceful_terminate_process(
    process: Optional[subprocess.Popen],
    timeout: float = 30
) -> Tuple[bool, Optional[int]]:
    """
    Gracefully terminate a process by sending an interrupt signal first,
    then falling back to forceful termination if it doesn't exit in time.

    On Windows, sends CTRL_BREAK_EVENT (requires CREATE_NEW_PROCESS_GROUP flag).
    On Unix, sends SIGINT.

    Args:
        process: subprocess.Popen object (must be created with CREATE_NEW_PROCESS_GROUP on Windows)
        timeout: Seconds to wait for graceful shutdown before force killing (default: 30)

    Returns:
        tuple: (graceful: bool, return_code: int or None)
            - graceful: True if process exited gracefully, False if force-killed
            - return_code: Process exit code, or None if unknown
    """
    if process is None:
        return True, None

    # Check if already exited
    if process.poll() is not None:
        return True, process.returncode

    import signal

    # Send interrupt signal
    if sys.platform == 'win32':
        try:
            # CTRL_BREAK_EVENT can be sent to processes in a different process group
            # The subprocess must have been created with CREATE_NEW_PROCESS_GROUP
            os.kill(process.pid, signal.CTRL_BREAK_EVENT)
        except (OSError, ProcessLookupError) as e:
            # Process may have already exited
            if process.poll() is not None:
                return True, process.returncode
            # Otherwise fall through to force termination
            terminate_process_tree(process)
            return False, process.poll()
    else:
        try:
            # On Unix, send SIGINT to the process group
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            if process.poll() is not None:
                return True, process.returncode
            terminate_process_tree(process)
            return False, process.poll()

    # Wait for graceful exit
    try:
        return_code = process.wait(timeout=timeout)
        return True, return_code
    except subprocess.TimeoutExpired:
        # Graceful shutdown timed out, force kill
        terminate_process_tree(process)
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return_code = None
        return False, return_code
