"""Platform-specific utilities."""

import sys

class PlatformHelper:
    """Helper for platform-specific functionality."""
    
    @staticmethod
    def is_windows():
        return sys.platform == 'win32'
    
    @staticmethod
    def is_mac():
        return sys.platform == 'darwin'
    
    @staticmethod
    def is_linux():
        return sys.platform.startswith('linux')
    
    @staticmethod
    def get_platform():
        return sys.platform
    
    @staticmethod
    def get_mousewheel_delta(event):
        """Get normalized mousewheel delta for current platform."""
        if PlatformHelper.is_windows():
            return int(-1 * (event.delta / 120))
        elif PlatformHelper.is_mac():
            return int(-1 * event.delta)
        else:  # Linux
            if event.num == 4:
                return -1
            elif event.num == 5:
                return 1
            return 0
        
    @staticmethod
    def bind_mousewheel(widget, callback):
        """Bind mousewheel events in a platform-specific way."""
        if PlatformHelper.is_windows() or PlatformHelper.is_mac():
            widget.bind("<MouseWheel>", callback)
        else:  # Linux
            widget.bind("<Button-4>", callback)
            widget.bind("<Button-5>", callback)