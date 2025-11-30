# src/memory_utils.py

import psutil
import logging

class MemoryMonitor:
    """
    Monitors system memory usage and stops execution if threshold exceeded.
    
    Parameters:
    -----------
        threshold : float
            Memory usage percentage threshold (0-100). Default: 80.0
        check_interval : int
            Check memory every N operations. Default: 10
        
    Example:
    --------
    >>> monitor = MemoryMonitor(threshold=80.0)
    >>> monitor.check()  # Raises MemoryError if > 80%
    """
    def __init__(self, threshold: float = 80.0, check_interval: int = 10):
        self.threshold = threshold
        self.check_interval = check_interval
        self.check_count = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def check(self, force: bool = False) -> float:
        """
        Check current memory usage.
        
        Parameters:
        -----------
            force : bool
                If True, check immediately regardless of interval.
                
        Returns:
        --------
            float : Current memory usage percentage
            
        Raises:
        -------
            MemoryError : If memory usage exceeds threshold
        """
        self.check_count += 1
        
        # Only check at intervals (for performance) unless forced
        if not force and self.check_count % self.check_interval != 0:
            return -1  # Signal that check was skipped
        
        memory = psutil.virtual_memory()
        usage_percent = memory.percent
        
        self.logger.info(
            f"Memory: {usage_percent:.1f}% "
            f"({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)"
        )
        
        # ✅ THIS IS THE CRITICAL PART - Make sure it actually raises!
        if usage_percent > self.threshold:
            error_msg = (
                f"Memory usage ({usage_percent:.1f}%) exceeded threshold ({self.threshold}%). "
                f"Used: {memory.used / (1024**3):.1f}GB / Total: {memory.total / (1024**3):.1f}GB"
            )
            self.logger.error(error_msg)
            raise MemoryError(error_msg)
        
        return usage_percent
    
    def get_current_usage(self) -> dict:
        """Get detailed memory statistics without checking threshold."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'used_gb': memory.used / (1024**3),
            'available_gb': memory.available / (1024**3),
            'total_gb': memory.total / (1024**3)
        }
