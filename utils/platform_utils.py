#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Platform-specific utilities.
"""

import platform
import tkinter as tk
from typing import Callable


class PlatformHelper:
    """Platform-specific helper functions."""
    
    @staticmethod
    def get_system() -> str:
        """Get current operating system.
        
        Returns:
            'Darwin' (macOS), 'Linux', or 'Windows'
        """
        return platform.system()
    
    @staticmethod
    def bind_mousewheel(canvas: tk.Canvas, 
                       on_enter: Callable,
                       on_leave: Callable) -> None:
        """Bind platform-specific mousewheel events.
        
        Args:
            canvas: Canvas widget to bind to
            on_enter: Callback when mouse enters
            on_leave: Callback when mouse leaves
        """
        canvas.bind("<Enter>", on_enter)
        canvas.bind("<Leave>", on_leave)
    
    @staticmethod
    def get_mousewheel_delta(event) -> int:
        """Get mousewheel scroll delta (platform-independent).
        
        Args:
            event: Tkinter mouse event
            
        Returns:
            Scroll delta (-1 or 1)
        """
        system = PlatformHelper.get_system()
        
        if system == "Darwin":  # macOS
            return -1 if event.delta > 0 else 1
        elif hasattr(event, 'num'):  # Linux
            return 1 if event.num == 5 or event.delta < 0 else -1
        else:  # Windows
            return -1 if event.delta > 0 else 1