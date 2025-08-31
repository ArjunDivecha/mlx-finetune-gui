#!/usr/bin/env python3
"""
Quick test script for adapter fusion experiments.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Add the current directory to Python path
    fusion_script = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui/fusion_adapters.py"
    
    # First, list available adapters
    print("=== Available Adapters ===")
    result = subprocess.run([sys.executable, fusion_script, "--list-adapters"], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("Error listing adapters:", result.stderr)
        return
    
    # Parse available adapters from output
    adapters = []
    lines = result.stdout.split('\n')
    for line in lines:
        if line.strip().startswith('- '):
            adapter_name = line.strip()[2:]  # Remove '- ' prefix
            adapters.append(adapter_name)
    
    if len(adapters) < 2:
        print("Need at least 2 adapters for fusion experiments")
        return
    
    print(f"Found {len(adapters)} adapters: {adapters}")
    
    # Test 1: Simple 50/50 fusion of first two adapters
    print("\n=== Test 1: 50/50 Weighted Average ===")
    adapter1, adapter2 = adapters[0], adapters[1]
    
    cmd = [
        sys.executable, fusion_script,
        "--adapters", adapter1, adapter2,
        "--weights", "0.5", "0.5",
        "--output-dir", "./fusion_experiments/test1_50_50",
        "--output-name", f"{adapter1}_{adapter2}_50_50"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Test 2: 70/30 fusion
    print("\n=== Test 2: 70/30 Weighted Average ===")
    cmd = [
        sys.executable, fusion_script,
        "--adapters", adapter1, adapter2,
        "--weights", "0.7", "0.3",
        "--output-dir", "./fusion_experiments/test2_70_30",
        "--output-name", f"{adapter1}_{adapter2}_70_30"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Test 3: SLERP fusion (if available)
    print("\n=== Test 3: SLERP Interpolation ===")
    cmd = [
        sys.executable, fusion_script,
        "--adapters", adapter1, adapter2,
        "--weights", "0.0", "0.5",  # t=0.5 for SLERP
        "--method", "slerp",
        "--output-dir", "./fusion_experiments/test3_slerp_50",
        "--output-name", f"{adapter1}_{adapter2}_slerp_50"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Test 4: Three-way fusion if we have enough adapters
    if len(adapters) >= 3:
        print("\n=== Test 4: Three-way Equal Fusion ===")
        adapter3 = adapters[2]
        cmd = [
            sys.executable, fusion_script,
            "--adapters", adapter1, adapter2, adapter3,
            "--weights", "0.33", "0.33", "0.34",
            "--output-dir", "./fusion_experiments/test4_three_way",
            "--output-name", f"{adapter1}_{adapter2}_{adapter3}_equal"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
    
    print("\n=== Fusion Tests Complete ===")
    print("Check the ./fusion_experiments/ directory for results!")

if __name__ == "__main__":
    main()