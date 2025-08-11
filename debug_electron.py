#!/usr/bin/env python3
"""
Debug script to test Electron app display using Playwright
"""

import asyncio
import subprocess
import time
import requests
import os
import signal
from playwright.async_api import async_playwright

class ElectronDebugger:
    def __init__(self):
        self.electron_process = None
        self.backend_url = "http://localhost:8000"
        self.app_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui"
        
    async def start_electron_app(self):
        """Start the Electron app"""
        try:
            print("🚀 Starting Electron app...")
            os.chdir(self.app_dir)
            
            # Start the app
            self.electron_process = subprocess.Popen(
                ["npm", "start"],
                cwd=self.app_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for backend to start
            print("⏳ Waiting for backend to start...")
            max_attempts = 30
            for i in range(max_attempts):
                try:
                    response = requests.get(f"{self.backend_url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ Backend started after {i+1} attempts")
                        return True
                except requests.exceptions.RequestException:
                    await asyncio.sleep(1)
                    continue
            
            print("❌ Backend failed to start within 30 seconds")
            return False
            
        except Exception as e:
            print(f"❌ Failed to start Electron app: {e}")
            return False
    
    def kill_electron_app(self):
        """Kill the Electron app"""
        if self.electron_process:
            print("🔪 Killing Electron app...")
            try:
                # Kill the process group
                os.killpg(os.getpgid(self.electron_process.pid), signal.SIGTERM)
            except:
                try:
                    self.electron_process.terminate()
                except:
                    self.electron_process.kill()
            self.electron_process = None
    
    async def debug_with_playwright(self):
        """Use Playwright to debug the Electron app"""
        async with async_playwright() as p:
            print("🎭 Starting Playwright debugging...")
            
            # Try to connect to Electron app
            try:
                # Check if there are any Electron windows
                print("🔍 Looking for Electron windows...")
                
                browser = await p.chromium.connect_over_cdp("http://localhost:9222")
                contexts = browser.contexts
                
                print(f"Found {len(contexts)} browser contexts")
                
                for i, context in enumerate(contexts):
                    pages = context.pages
                    print(f"Context {i} has {len(pages)} pages")
                    
                    for j, page in enumerate(pages):
                        print(f"  Page {j}: {page.url}")
                        
                        # Try to get page title and content
                        try:
                            title = await page.title()
                            print(f"    Title: {title}")
                            
                            # Take screenshot if possible
                            screenshot_path = f"/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Fine Tuner/mlx-finetune-gui/debug_screenshot_{i}_{j}.png"
                            await page.screenshot(path=screenshot_path)
                            print(f"    Screenshot saved: {screenshot_path}")
                            
                        except Exception as e:
                            print(f"    Error accessing page: {e}")
                
                await browser.close()
                
            except Exception as e:
                print(f"❌ Failed to connect via CDP: {e}")
                
            # Try launching Electron directly with Playwright
            try:
                print("🚀 Trying to launch Electron with Playwright...")
                
                # Launch Electron app
                from playwright.async_api import BrowserType
                electron_executable = f"{self.app_dir}/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron"
                if not os.path.exists(electron_executable):
                    electron_executable = f"{self.app_dir}/node_modules/.bin/electron"
                
                electron_app = await p.electron.launch(
                    executable_path=electron_executable,
                    args=["."],
                    cwd=self.app_dir
                )
                
                print("✅ Electron launched with Playwright")
                
                # Get the main window
                main_window = await electron_app.first_window()
                print(f"📱 Main window: {main_window}")
                
                # Get window properties
                title = await main_window.title()
                url = main_window.url
                print(f"Title: {title}")
                print(f"URL: {url}")
                
                # Take screenshot
                screenshot_path = f"{self.app_dir}/debug_main_window.png"
                await main_window.screenshot(path=screenshot_path)
                print(f"📸 Screenshot saved: {screenshot_path}")
                
                # Wait a bit to see if content loads
                await asyncio.sleep(3)
                
                # Check if content is loaded
                body_content = await main_window.content()
                print(f"📄 Body content length: {len(body_content)}")
                
                if "MLX" in body_content or "fine" in body_content.lower():
                    print("✅ App content appears to be loaded")
                else:
                    print("⚠️ App content may not be loaded properly")
                    print(f"First 500 chars: {body_content[:500]}")
                
                # Check for errors in console
                main_window.on("console", lambda msg: print(f"Console: {msg.text}"))
                main_window.on("pageerror", lambda error: print(f"Page Error: {error}"))
                
                # Wait for any async loading
                await asyncio.sleep(5)
                
                # Take final screenshot
                final_screenshot_path = f"{self.app_dir}/debug_final.png"
                await main_window.screenshot(path=final_screenshot_path)
                print(f"📸 Final screenshot: {final_screenshot_path}")
                
                # Close Electron app
                await electron_app.close()
                print("🔚 Electron app closed")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to launch with Playwright: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    async def test_backend_directly(self):
        """Test backend endpoints directly"""
        print("🧪 Testing backend endpoints...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.backend_url}/health")
            print(f"Health endpoint: {response.status_code}")
            
            # Test status endpoint
            response = requests.get(f"{self.backend_url}/training/status")
            print(f"Status endpoint: {response.status_code} - {response.json()}")
            
            # Test sessions endpoint
            response = requests.get(f"{self.backend_url}/sessions")
            print(f"Sessions endpoint: {response.status_code} - {response.json()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Backend test failed: {e}")
            return False
    
    async def check_frontend_files(self):
        """Check if frontend files exist and are valid"""
        print("📁 Checking frontend files...")
        
        frontend_dist = f"{self.app_dir}/frontend/dist"
        
        # Check if dist folder exists
        if not os.path.exists(frontend_dist):
            print(f"❌ Frontend dist folder missing: {frontend_dist}")
            return False
        
        # Check for index.html
        index_html = f"{frontend_dist}/index.html"
        if not os.path.exists(index_html):
            print(f"❌ index.html missing: {index_html}")
            return False
        
        # Check index.html content
        with open(index_html, 'r') as f:
            content = f.read()
            if len(content) < 100:
                print(f"⚠️ index.html seems too small: {len(content)} chars")
            else:
                print(f"✅ index.html exists: {len(content)} chars")
        
        # Check assets folder
        assets_dir = f"{frontend_dist}/assets"
        if os.path.exists(assets_dir):
            assets = os.listdir(assets_dir)
            print(f"✅ Assets folder contains: {len(assets)} files")
            for asset in assets[:5]:  # Show first 5
                print(f"  - {asset}")
        else:
            print("❌ Assets folder missing")
            return False
        
        return True
    
    async def run_comprehensive_debug(self):
        """Run comprehensive debugging"""
        print("🔍 Starting comprehensive Electron debugging...")
        print("=" * 60)
        
        # 1. Check frontend files
        frontend_ok = await self.check_frontend_files()
        if not frontend_ok:
            print("🛠️ Building frontend...")
            result = subprocess.run(["npm", "run", "build:frontend"], 
                                  cwd=self.app_dir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Frontend build failed: {result.stderr}")
                return False
            print("✅ Frontend built successfully")
        
        # 2. Start Electron app
        app_started = await self.start_electron_app()
        if not app_started:
            print("❌ Failed to start Electron app")
            return False
        
        # 3. Test backend
        backend_ok = await self.test_backend_directly()
        if not backend_ok:
            print("❌ Backend not responding properly")
        
        # 4. Debug with Playwright
        playwright_ok = await self.debug_with_playwright()
        
        # 5. Cleanup
        self.kill_electron_app()
        
        print("=" * 60)
        if playwright_ok:
            print("✅ Debugging completed successfully - check screenshots")
        else:
            print("❌ Debugging found issues - check error messages above")
        
        return playwright_ok

async def main():
    debugger = ElectronDebugger()
    await debugger.run_comprehensive_debug()

if __name__ == "__main__":
    asyncio.run(main())