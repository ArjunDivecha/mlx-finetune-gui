#!/usr/bin/env python3
"""
Test script to compare investment advice from different adapter versions.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from datetime import datetime

def test_inference_with_adapter(adapter_path, prompt, model_path):
    """Test inference with a specific adapter."""
    python_path = '/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python'
    
    if adapter_path:
        # With adapter
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}", adapter_path="{adapter_path}")
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
    print("RESPONSE_START")
    print(response)
    print("RESPONSE_END")
except Exception as e:
    print(f"Error: {{e}}")
    import traceback
    traceback.print_exc()
'''
        ]
    else:
        # Base model only
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

try:
    model, tokenizer = load("{model_path}")
    prompt = """{prompt}"""
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300)
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
    # Investment question
    investment_prompt = "What are the benefits of investing in emerging markets?"
    
    # Base model path
    base_model = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model/Qwen3-30B-A3B-Instruct-2507-4bit"
    adapter_base_path = "/Users/macbook2024/Library/CloudStorage/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
    
    # Check if base model exists
    if not os.path.exists(base_model):
        print("Base model not found. Please check the path.")
        return
    
    # Test configurations
    test_configs = [
        {
            "name": "Original Adapter 1 (mlx_finetune_new)",
            "adapter_path": os.path.join(adapter_base_path, "mlx_finetune_new"),
            "description": "First original adapter from your training runs"
        },
        {
            "name": "Original Adapter 2 (mlx_finetune_new2)", 
            "adapter_path": os.path.join(adapter_base_path, "mlx_finetune_new2"),
            "description": "Second original adapter from your training runs"
        },
        {
            "name": "50/50 Fusion",
            "adapter_path": "./fusion_test",
            "description": "Equal blend of both original adapters"
        },
        {
            "name": "70/30 Fusion (favoring first)",
            "adapter_path": "./fusion_70_30", 
            "description": "70% first adapter, 30% second adapter"
        },
        {
            "name": "SLERP Fusion",
            "adapter_path": "./fusion_slerp",
            "description": "Spherical interpolation between adapters"
        }
    ]
    
    results = []
    
    print("💼 Investment Advice Comparison")
    print("=" * 60)
    print(f"Question: {investment_prompt}")
    print("=" * 60)
    
    # Test each configuration
    for i, config in enumerate(test_configs, 1):
        print(f"\n🔬 Test {i}/5: {config['name']}")
        print(f"Description: {config['description']}")
        
        if not os.path.exists(config['adapter_path']):
            print(f"⚠️ Adapter path not found: {config['adapter_path']}")
            continue
        
        print("Generating response...")
        response = test_inference_with_adapter(config['adapter_path'], investment_prompt, base_model)
        
        result = {
            "name": config['name'],
            "description": config['description'],
            "adapter_path": config['adapter_path'], 
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        
        results.append(result)
        
        # Show preview
        preview = response[:200] if len(response) > 200 else response
        print(f"Response preview: {preview}...")
        print("-" * 40)
    
    # Save detailed results
    output_file = "investment_advice_comparison.json"
    with open(output_file, "w") as f:
        json.dump({
            "question": investment_prompt,
            "test_date": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    
    print(f"\n✅ Testing complete! Detailed results saved to {output_file}")
    
    # Create readable summary
    summary_file = "investment_advice_summary.txt"
    with open(summary_file, "w") as f:
        f.write("INVESTMENT ADVICE COMPARISON\n")
        f.write("=" * 50 + "\n")
        f.write(f"Question: {investment_prompt}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"{i}. {result['name']}\n")
            f.write(f"   {result['description']}\n")
            f.write(f"   Response:\n")
            f.write(f"   {result['response']}\n")
            f.write("-" * 50 + "\n\n")
    
    print(f"📝 Human-readable summary saved to {summary_file}")
    
    # Show quick comparison
    print("\n📊 QUICK COMPARISON:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['name']}")
        preview = result['response'][:150] if len(result['response']) > 150 else result['response']
        print(f"   {preview}...")

if __name__ == "__main__":
    main()