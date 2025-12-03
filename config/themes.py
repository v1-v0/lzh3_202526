"""Theme management for the application."""

class ThemeManager:
    """Manages UI themes and styling."""
    
    def __init__(self):
        self.current_theme = "default"
        self.colors = {
            'bg': '#ffffff',
            'fg': '#000000',
            'accent': '#0078d4',
        }
        self.dark_colors = {
            'bg': '#1e1e1e',
            'fg': '#ffffff',
            'accent': '#0078d4',
        }
        self.light_colors = {
            'bg': '#ffffff',
            'fg': '#000000',
            'accent': '#0078d4',
        }
    
    def apply_theme(self, widget):
        """Apply theme to a widget."""
        pass
    
    def apply_dark_theme(self, root):
        """Apply dark theme to the application."""
        self.current_theme = "dark"
        self.colors = self.dark_colors.copy()
        
        # Apply to root window
        root.configure(bg=self.dark_colors['bg'])
        
        # You can extend this to update all widgets
        for widget in root.winfo_children():
            self._apply_dark_to_widget(widget)
    
    def apply_light_theme(self, root):
        """Apply light theme to the application."""
        self.current_theme = "light"
        self.colors = self.light_colors.copy()
        
        # Apply to root window
        root.configure(bg=self.light_colors['bg'])
        
        # Update all widgets
        for widget in root.winfo_children():
            self._apply_light_to_widget(widget)
    
    def _apply_dark_to_widget(self, widget):
        """Recursively apply dark theme to widget and children."""
        try:
            widget.configure(bg=self.dark_colors['bg'], fg=self.dark_colors['fg'])
        except:
            pass
        
        for child in widget.winfo_children():
            self._apply_dark_to_widget(child)
    
    def _apply_light_to_widget(self, widget):
        """Recursively apply light theme to widget and children."""
        try:
            widget.configure(bg=self.light_colors['bg'], fg=self.light_colors['fg'])
        except:
            pass
        
        for child in widget.winfo_children():
            self._apply_light_to_widget(child)
    
    def get_color(self, name: str) -> str:
        """Get color from current theme."""
        return self.colors.get(name, '#000000')