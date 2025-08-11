#!/usr/bin/env python3
"""
Script to find and show the MLX Fine-tuning GUI window
"""

import subprocess
import time
import os

def find_and_show_window():
    """Find the MLX GUI window and bring it to front"""
    
    print("🔍 Looking for MLX Fine-tuning GUI window...")
    
    try:
        # First, try to get all windows with detailed info
        script = '''
        tell application "System Events"
            set appList to every application process whose visible is true
            repeat with appProcess in appList
                try
                    set appName to name of appProcess
                    if appName contains "Electron" then
                        set windowList to every window of appProcess
                        repeat with aWindow in windowList
                            try
                                set windowTitle to name of aWindow
                                set windowPosition to position of aWindow
                                set windowSize to size of aWindow
                                log "App: " & appName & ", Window: " & windowTitle & ", Pos: " & (item 1 of windowPosition) & "," & (item 2 of windowPosition) & ", Size: " & (item 1 of windowSize) & "x" & (item 2 of windowSize)
                                
                                -- Check if this looks like our app window
                                if windowTitle contains "MLX" or windowTitle = "" or windowTitle contains "Electron" then
                                    -- Try to bring to front
                                    set frontmost of appProcess to true
                                    set aWindow to front window of appProcess
                                    log "Brought window to front: " & windowTitle
                                    return true
                                end if
                            on error errMsg
                                log "Error with window: " & errMsg
                            end try
                        end repeat
                    end if
                on error errMsg
                    log "Error with app: " & errMsg
                end try
            end repeat
        end tell
        '''
        
        result = subprocess.run([
            'osascript', '-e', script
        ], capture_output=True, text=True)
        
        if result.stderr:
            print("📋 AppleScript output:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        # Also try to activate any Electron app that might be our GUI
        print("🎯 Attempting to activate MLX GUI...")
        
        # Try multiple approaches
        approaches = [
            'tell application "Electron" to activate',
            'tell application "System Events" to tell application process "Electron" to set frontmost to true',
            '''
            tell application "System Events"
                set frontmost of first application process whose name is "Electron" to true
            end tell
            '''
        ]
        
        for i, approach in enumerate(approaches, 1):
            try:
                subprocess.run(['osascript', '-e', approach], 
                             capture_output=True, timeout=5)
                print(f"✅ Tried approach {i}")
                time.sleep(1)
            except:
                print(f"❌ Approach {i} failed")
        
        # Check if we have any windows positioned off-screen
        print("🔍 Checking for off-screen windows...")
        
        off_screen_script = '''
        tell application "System Events"
            repeat with appProcess in (every application process whose visible is true)
                try
                    if name of appProcess contains "Electron" then
                        repeat with aWindow in (every window of appProcess)
                            try
                                set windowPos to position of aWindow
                                set windowSize to size of aWindow
                                set x to item 1 of windowPos
                                set y to item 2 of windowPos
                                set w to item 1 of windowSize  
                                set h to item 2 of windowSize
                                
                                -- Check if window is off-screen (negative coords or too far)
                                if x < -1000 or y < -1000 or x > 3000 or y > 3000 then
                                    log "Off-screen window found at " & x & "," & y & " size " & w & "x" & h
                                    -- Try to move it to visible area
                                    set position of aWindow to {100, 100}
                                    log "Moved window to 100,100"
                                end if
                            end try
                        end repeat
                    end if
                end try
            end repeat
        end tell
        '''
        
        result = subprocess.run(['osascript', '-e', off_screen_script],
                               capture_output=True, text=True)
        
        if result.stderr:
            print("📋 Off-screen check output:")
            for line in result.stderr.split('\n'):
                if line.strip():
                    print(f"  {line}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_backend_connection():
    """Test if the backend is responding"""
    print("🧪 Testing backend connection...")
    
    import urllib.request
    import json
    
    try:
        # Test health endpoint
        with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
            if response.status == 200:
                print("✅ Backend health check passed")
            else:
                print(f"⚠️ Backend health check returned {response.status}")
                
        # Test status endpoint  
        with urllib.request.urlopen('http://localhost:8000/training/status', timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"✅ Training status: {data.get('state', 'unknown')}")
            else:
                print(f"⚠️ Training status returned {response.status}")
                
        return True
        
    except Exception as e:
        print(f"❌ Backend connection failed: {e}")
        return False

def main():
    print("🔧 MLX Fine-tuning GUI Window Finder")
    print("=" * 50)
    
    # Test backend first
    backend_ok = test_backend_connection()
    
    if not backend_ok:
        print("❌ Backend not responding - GUI may not work properly")
    
    print()
    
    # Try to find and show the window
    success = find_and_show_window()
    
    if success:
        print("✅ Window search completed")
        print("\n💡 Tips:")
        print("  - Try Cmd+Tab to cycle through applications")  
        print("  - Look for 'Electron' in the application switcher")
        print("  - Check Activity Monitor for 'Electron' processes")
        print("  - The app may be minimized to the dock")
        print("  - Try clicking the Electron icon in the dock")
    else:
        print("❌ Could not locate window")
    
    print("=" * 50)

if __name__ == "__main__":
    main()