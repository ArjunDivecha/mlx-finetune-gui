#!/usr/bin/env python3
"""
Test script for comparing responses from different fused adapters.
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def test_inference_with_adapter(adapter_path, prompt, model_path):
    """Test inference with a specific adapter."""
    python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python'
    
    cmd = [
        python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}", adapter_path="{adapter_path}")
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
    print("RESPONSE_START")
    print(response)
    print("RESPONSE_END")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''
    ]
    
    try:
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if process.returncode != 0:
            return f"Error: {process.stderr}"
        
        # Extract response
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            start_idx = output.find("RESPONSE_START") + len("RESPONSE_START")
            end_idx = output.find("RESPONSE_END")
            response = output[start_idx:end_idx].strip()
        else:
            response = output.strip()
        
        return response
    
    except subprocess.TimeoutExpired:
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {e}"

def main():
    # Test prompt
    test_prompt = "Write a short poem about artificial intelligence:"
    
    # Base model path (update if different)
    base_model = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-30B-A3B-Instruct-2507-4bit"
    
    # Check if base model exists
    if not os.path.exists(base_model):
        print("Base model not found. Please check the path.")
        return
    
    # Fusion experiments to test
    fusion_dirs = [
        "./fusion_test",      # 50/50 weighted
        "./fusion_70_30",     # 70/30 weighted  
        "./fusion_slerp"      # SLERP interpolation
    ]
    
    results = {}
    
    print("🤖 Testing Fused Adapters")
    print("=" * 50)
    print(f"Test prompt: {test_prompt}")
    print("=" * 50)
    
    # Test base model first (no adapter)
    print("\n📝 Testing Base Model (No Adapter)...")
    base_response = test_inference_with_adapter("", test_prompt, base_model)
    results["base_model"] = base_response
    print(f"Response: {base_response[:200]}...")
    
    # Test each fusion
    for fusion_dir in fusion_dirs:
        if os.path.exists(fusion_dir):
            adapter_path = fusion_dir
            fusion_name = os.path.basename(fusion_dir)
            
            print(f"\n🔀 Testing {fusion_name}...")
            response = test_inference_with_adapter(adapter_path, test_prompt, base_model)
            results[fusion_name] = response
            print(f"Response: {response[:200]}...")
        else:
            print(f"\n⚠️ Fusion directory not found: {fusion_dir}")
    
    # Save results
    with open("fusion_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Testing complete! Results saved to fusion_test_results.json")
    
    # Show comparison
    print("\n📊 Quick Comparison:")
    print("-" * 50)
    for name, response in results.items():
        print(f"{name}: {response[:100]}...")
        print()

if __name__ == "__main__":
    main()