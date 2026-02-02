"""
Safe configuration file manager using AST parsing
Replaces the fragile string-matching approach in feedback_tuner.py
"""

import ast
import astor  # Install: pip install astor
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import fields


class ConfigFileManager:
    """Manages bacteria_configs.py using AST parsing"""
    
    def __init__(self, config_file: Path):
        """
        Initialize the config file manager
        
        Args:
            config_file: Path to bacteria_configs.py
        """
        self.config_file = config_file
        self.tree: Optional[ast.Module] = None  # ← Type hint added
        self.source: Optional[str] = None
        
    def load(self) -> bool:
        """Load and parse the config file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.source = f.read()
            
            self.tree = ast.parse(self.source)
            return True
        except Exception as e:
            print(f"❌ Failed to load config file: {e}")
            return False
    
    def find_config_assignment(self, var_name: str) -> Optional[Tuple[int, ast.Assign]]:
        """Find the assignment node for a specific config variable
        
        Args:
            var_name: Variable name (e.g., 'PROTEUS_MIRABILIS')
            
        Returns:
            Tuple of (node_index, assignment_node) or None if not found
        """
        if self.tree is None:  # ← Fixed type check
            return None
        
        for idx, node in enumerate(self.tree.body):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        return idx, node
        
        return None
    
    def create_config_assignment(self, var_name: str, config_data: dict) -> ast.Assign:
        """Create an AST assignment node for a SegmentationConfig
        
        Args:
            var_name: Variable name (e.g., 'PROTEUS_MIRABILIS')
            config_data: Dictionary of configuration parameters
            
        Returns:
            ast.Assign node
        """
        # Create keyword arguments for SegmentationConfig
        keywords = []
        
        for key, value in config_data.items():
            # Convert value to appropriate AST node
            if isinstance(value, str):
                value_node = ast.Constant(value=value)
            elif isinstance(value, (int, float)):
                value_node = ast.Constant(value=value)
            elif isinstance(value, bool):
                value_node = ast.Constant(value=value)
            else:
                value_node = ast.Constant(value=value)
            
            keywords.append(ast.keyword(arg=key, value=value_node))
        
        # Create SegmentationConfig(...) call
        config_call = ast.Call(
            func=ast.Name(id='SegmentationConfig', ctx=ast.Load()),
            args=[],
            keywords=keywords
        )
        
        # Create assignment: VAR_NAME = SegmentationConfig(...)
        assignment = ast.Assign(
            targets=[ast.Name(id=var_name, ctx=ast.Store())],
            value=config_call
        )
        
        return assignment
    
    def update_config(self, var_name: str, config_data: dict) -> bool:
        """Update or add a configuration
        
        Args:
            var_name: Variable name (e.g., 'PROTEUS_MIRABILIS')
            config_data: Dictionary of configuration parameters
            
        Returns:
            True if successful, False otherwise
        """
        if self.tree is None:  # ← Fixed type check
            if not self.load():
                return False
        
        # Assert for type checker (we know tree is not None after load())
        assert self.tree is not None  # ← Type narrowing
        
        # Create new assignment node
        new_node = self.create_config_assignment(var_name, config_data)
        
        # Find existing assignment
        result = self.find_config_assignment(var_name)
        
        if result:
            # Replace existing
            idx, _ = result
            self.tree.body[idx] = new_node
            print(f"  ✓ Updated existing {var_name} configuration")
        else:
            # Insert before DEFAULT (or append if DEFAULT not found)
            default_idx = self._find_default_config_index()
            
            if default_idx is not None:
                self.tree.body.insert(default_idx, new_node)
                print(f"  ✓ Inserted new {var_name} configuration before DEFAULT")
            else:
                self.tree.body.append(new_node)
                print(f"  ✓ Appended new {var_name} configuration")
        
        return True
    
    def _find_default_config_index(self) -> Optional[int]:
        """Find the index of DEFAULT config assignment
        
        Returns:
            Index or None if not found
        """
        result = self.find_config_assignment('DEFAULT')
        return result[0] if result else None
    
    def save(self, backup: bool = True) -> bool:
        """Save the modified AST back to file
        
        Args:
            backup: Create .bak backup before saving
            
        Returns:
            True if successful, False otherwise
        """
        if self.tree is None:  # ← Fixed type check
            print("❌ No AST tree to save")
            return False
        
        try:
            # Create backup
            if backup:
                backup_path = self.config_file.with_suffix('.py.bak')
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    with open(backup_path, 'w', encoding='utf-8') as bf:
                        bf.write(f.read())
                print(f"  ✓ Created backup: {backup_path.name}")
            
            # Generate source code from AST
            new_source = astor.to_source(self.tree)
            
            # Write to file
            with open(self.config_file, 'w', encoding='utf-8') as f:
                f.write(new_source)
            
            print(f"  ✓ Saved to {self.config_file.name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save: {e}")
            return False
    
    def validate_syntax(self) -> bool:
        """Validate that the generated file has valid Python syntax
        
        Returns:
            True if valid, False otherwise
        """
        if self.tree is None:  # ← Fixed type check
            return False
        
        try:
            compile(astor.to_source(self.tree), str(self.config_file), 'exec')
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error in generated code: {e}")
            return False


def config_to_dict(config) -> dict:
    """Convert SegmentationConfig to dictionary
    
    Args:
        config: SegmentationConfig object
        
    Returns:
        Dictionary of config parameters
    """
    from bacteria_configs import SegmentationConfig
    
    if not isinstance(config, SegmentationConfig):
        raise TypeError("config must be a SegmentationConfig instance")
    
    # Get all fields from dataclass
    return {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }


# Example usage function
def update_bacteria_config(bacterium: str, config, backup: bool = True) -> bool:
    """Update bacteria configuration in bacteria_configs.py
    
    Args:
        bacterium: Bacterium name (e.g., "Proteus mirabilis")
        config: SegmentationConfig object
        backup: Create backup before modifying
        
    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path
    
    config_file = Path(__file__).parent / "bacteria_configs.py"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Create manager
    manager = ConfigFileManager(config_file)
    
    # Load file
    if not manager.load():
        return False
    
    # Convert bacterium name to variable name
    var_name = bacterium.upper().replace(' ', '_').replace('.', '')
    
    # Convert config to dict
    config_dict = config_to_dict(config)
    
    # Update config
    if not manager.update_config(var_name, config_dict):
        return False
    
    # Validate syntax
    if not manager.validate_syntax():
        print("❌ Generated code has syntax errors - not saving")
        return False
    
    # Save
    return manager.save(backup=backup)


if __name__ == "__main__":
    # Demo/test
    print("🧪 Testing ConfigFileManager...")
    
    from bacteria_configs import SegmentationConfig
    
    test_config = SegmentationConfig(
        name="Test Bacterium",
        description="Test configuration for AST parser",
        gaussian_sigma=5.0,
        min_area_um2=1.0,
        max_area_um2=100.0,
    )
    
    success = update_bacteria_config(
        bacterium="Test Bacterium",
        config=test_config,
        backup=True
    )
    
    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")