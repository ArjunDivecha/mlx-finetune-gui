#!/usr/bin/env python3
"""
Simple debug script to check Electron app issues
"""

import subprocess
import time
import os
import requests
import signal
from pathlib import Path

def check_frontend_build():
    """Check if frontend is properly built"""
    print("🔍 Checking frontend build...")
    
    app_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui"
    frontend_dist = f"{app_dir}/frontend/dist"
    
    if not os.path.exists(frontend_dist):
        print("❌ Frontend dist folder missing - building...")
        result = subprocess.run(["npm", "run", "build:frontend"], cwd=app_dir)
        return result.returncode == 0
    
    index_html = f"{frontend_dist}/index.html"
    if not os.path.exists(index_html):
        print("❌ index.html missing - rebuilding...")
        result = subprocess.run(["npm", "run", "build:frontend"], cwd=app_dir)
        return result.returncode == 0
    
    print("✅ Frontend build exists")
    return True

def check_main_build():
    """Check if main.js is built"""
    print("🔍 Checking main.js build...")
    
    app_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui"
    main_js = f"{app_dir}/dist/main.js"
    
    if not os.path.exists(main_js):
        print("❌ main.js missing - building...")
        result = subprocess.run(["npm", "run", "build:main"], cwd=app_dir)
        return result.returncode == 0
    
    print("✅ main.js exists")
    return True

def test_electron_launch():
    """Test launching Electron and check for errors"""
    print("🚀 Testing Electron launch...")
    
    app_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui"
    
    try:
        # Start Electron process
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=app_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("⏳ Waiting for Electron to start...")
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Electron process is running")
            
            # Try to connect to backend
            print("🔗 Testing backend connection...")
            max_attempts = 10
            backend_connected = False
            
            for i in range(max_attempts):
                try:
                    response = requests.get("http://localhost:8000/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Backend connected after {i+1} attempts")
                        backend_connected = True
                        break
                except:
                    time.sleep(1)
            
            if not backend_connected:
                print("❌ Backend not responding")
            
            # Check for visible windows
            print("🔍 Checking for visible windows...")
            try:
                result = subprocess.run([
                    "osascript", "-e", 
                    'tell application "System Events" to get name of every application process whose visible is true'
                ], capture_output=True, text=True)
                
                if "Electron" in result.stdout:
                    print("✅ Electron app is visible")
                else:
                    print("❌ Electron app not visible")
                    print("Visible apps:", result.stdout)
                    
            except Exception as e:
                print(f"❌ Error checking visibility: {e}")
            
            # Try to bring to front
            print("🎯 Attempting to bring Electron to front...")
            try:
                subprocess.run([
                    "osascript", "-e", 
                    'tell application "Electron" to activate'
                ])
                print("✅ Attempted to activate Electron")
            except:
                pass
            
            # Wait a bit more
            time.sleep(3)
            
            # Get stderr output
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                stdout, stderr = process.communicate()
            
            if stderr:
                print("📋 Electron stderr output:")
                print(stderr[-1000:])  # Last 1000 chars
            
            if stdout:
                print("📋 Electron stdout output:")  
                print(stdout[-1000:])  # Last 1000 chars
                
        else:
            print("❌ Electron process exited")
            stdout, stderr = process.communicate()
            
            if stderr:
                print("Error output:", stderr)
            if stdout:
                print("Output:", stdout)
                
            return False
            
        # Kill process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Electron: {e}")
        return False

def check_display_environment():
    """Check if we have a proper display environment"""
    print("🖥️ Checking display environment...")
    
    # Check if we're in SSH
    if os.getenv('SSH_CONNECTION'):
        print("⚠️ Running via SSH - GUI apps may not display")
    else:
        print("✅ Not in SSH session")
    
    # Check display variable
    display = os.getenv('DISPLAY')
    if display:
        print(f"✅ DISPLAY set to: {display}")
    else:
        print("ℹ️ No DISPLAY variable (normal for macOS)")
    
    # Check if we can run GUI apps
    try:
        result = subprocess.run([
            "osascript", "-e", 
            'display dialog "Test" buttons {"OK"} default button 1 giving up after 1'
        ], capture_output=True, timeout=5)
        
        if result.returncode == 0:
            print("✅ Can display GUI dialogs")
        else:
            print("❌ Cannot display GUI dialogs")
            
    except Exception as e:
        print(f"❌ Error testing GUI: {e}")

def main():
    print("🔧 MLX Fine-tuning GUI Debug Tool")
    print("=" * 50)
    
    # Check display environment
    check_display_environment()
    print()
    
    # Check builds
    frontend_ok = check_frontend_build()
    main_ok = check_main_build()
    
    if not frontend_ok or not main_ok:
        print("❌ Build issues detected")
        return
    
    print()
    
    # Test Electron
    electron_ok = test_electron_launch()
    
    print()
    print("=" * 50)
    
    if electron_ok:
        print("✅ Debug completed - Electron should be visible")
        print("If you still can't see it, try:")
        print("  1. Check Activity Monitor for 'Electron' processes")
        print("  2. Try Cmd+Tab to cycle through applications")
        print("  3. Check if window is positioned off-screen")
    else:
        print("❌ Debug found issues with Electron startup")

if __name__ == "__main__":
    main()