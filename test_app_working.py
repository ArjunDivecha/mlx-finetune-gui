#!/usr/bin/env python3
"""
Quick test to verify the MLX Fine-tuning GUI is working
"""

import subprocess
import requests
import time
import os

def test_app():
    print("🧪 Testing MLX Fine-tuning GUI")
    print("=" * 50)
    
    # 1. Check if Electron process is running
    try:
        result = subprocess.run(["pgrep", "-f", "Electron"], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"✅ Electron processes running: {len(pids)} processes")
        else:
            print("❌ No Electron processes found")
            return False
    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return False
    
    # 2. Test backend connectivity
    try:
        response = requests.get("http://127.0.0.1:8000/training/status", timeout=5)
        if response.status_code == 200:
            print("✅ Backend responding correctly")
            status_data = response.json()
            print(f"   Training state: {status_data.get('state', 'unknown')}")
        else:
            print(f"❌ Backend returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False
    
    # 3. Check if window is visible
    try:
        result = subprocess.run([
            "osascript", "-e",
            'tell application "System Events" to get name of every application process whose visible is true'
        ], capture_output=True, text=True)
        
        if "Electron" in result.stdout:
            print("✅ Electron app is visible in system")
        else:
            print("❌ Electron app not visible")
            return False
    except Exception as e:
        print(f"❌ Error checking window visibility: {e}")
        return False
    
    # 4. Try to take a screenshot
    try:
        screenshot_path = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui/app_screenshot.png"
        
        subprocess.run([
            "screencapture", "-l$(osascript -e 'tell app \"System Events\" to tell process \"Electron\" to get window 1's id')", 
            screenshot_path
        ], capture_output=True)
        
        if os.path.exists(screenshot_path):
            print(f"✅ Screenshot saved: {screenshot_path}")
        else:
            print("ℹ️ Screenshot not captured (normal if window is off-screen)")
    except Exception as e:
        print(f"ℹ️ Could not capture screenshot: {e}")
    
    print("=" * 50)
    print("🎉 MLX Fine-tuning GUI appears to be working correctly!")
    print()
    print("📋 Summary:")
    print("  ✅ Electron desktop app is running")
    print("  ✅ Backend API is responding") 
    print("  ✅ Window is visible in system")
    print("  ✅ App should be accessible as a desktop application")
    print()
    print("💡 If you can't see the window:")
    print("  - Try Cmd+Tab to cycle through applications")
    print("  - Look for 'Electron' or 'MLX Fine-tuning GUI' in the app switcher")
    print("  - Check if the window is positioned off-screen")
    print("  - Click on the Electron icon in the dock")
    
    return True

if __name__ == "__main__":
    test_app()