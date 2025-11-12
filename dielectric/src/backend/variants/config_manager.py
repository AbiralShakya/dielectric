"""
Configuration Manager

Features:
- Design parameters
- Stackup configurations
- Design rule sets
- Template management
"""

import logging
import json
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration Manager for PCB designs.
    
    Features:
    - Design parameters
    - Stackup configurations
    - Design rule sets
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Dict = {}
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
    
    def save_config(self):
        """Save configuration to file."""
        if self.config_path:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def get_stackup_config(self) -> Dict:
        """Get layer stackup configuration."""
        return self.config.get("stackup", {
            "layers": [
                {"name": "F.Cu", "type": "signal", "thickness": 0.035},
                {"name": "dielectric", "type": "core", "thickness": 1.6, "er": 4.5},
                {"name": "B.Cu", "type": "signal", "thickness": 0.035}
            ]
        })
    
    def get_design_rules(self) -> Dict:
        """Get design rule set."""
        return self.config.get("design_rules", {
            "min_trace_width": 0.1,
            "min_trace_spacing": 0.1,
            "min_via_size": 0.5,
            "min_via_drill": 0.2,
            "min_copper_to_edge": 0.5
        })

