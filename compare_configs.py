#!/usr/bin/env python3
"""
Utility to compare different bacteria configurations
Usage: python compare_configs.py
"""

from bacteria_configs import print_config_comparison, CONFIGS

if __name__ == "__main__":
    print_config_comparison()
    
    print("\nAvailable configuration keys:")
    for key in sorted(CONFIGS.keys()):
        cfg = CONFIGS[key]
        print(f"  '{key}' → {cfg.name}")

