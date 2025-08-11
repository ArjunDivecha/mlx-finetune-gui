#!/usr/bin/env python3
"""
Playwright Training Monitor
Monitors the MLX Fine-Tuner training page for issues and generates detailed reports
"""

import asyncio
import json
import time
from datetime import datetime
from playwright.async_api import async_playwright
import sys

class TrainingMonitor:
    def __init__(self):
        self.page = None
        self.browser = None
        self.monitoring = False
        self.chart_data_points = []
        self.status_updates = []
        self.errors = []
        
    async def setup(self):
        """Setup Playwright browser and page"""
        print("🚀 Setting up Playwright monitoring...")
        playwright = await async_playwright().start()
        
        # Launch browser with dev tools for debugging
        self.browser = await playwright.chromium.launch(
            headless=False,
            devtools=True,
            args=['--disable-web-security', '--disable-features=VizDisplayCompositor']
        )
        
        # Create page and setup monitoring
        self.page = await self.browser.new_page()
        
        # Enable console logging
        self.page.on("console", self.handle_console)
        self.page.on("pageerror", self.handle_page_error)
        
        # Navigate to training page
        print("📱 Navigating to training page...")
        await self.page.goto("http://localhost:3000")
        
        # Wait for page to load and navigate to training
        await self.page.wait_for_selector('[data-testid="nav-training"], text="Training"', timeout=10000)
        await self.page.click('[data-testid="nav-training"], text="Training"')
        
        print("✅ Training page loaded, starting monitoring...")
        
    async def handle_console(self, msg):
        """Handle console messages"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"🖥️  [{timestamp}] CONSOLE {msg.type.upper()}: {msg.text}")
        
    async def handle_page_error(self, error):
        """Handle page errors"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"❌ [{timestamp}] PAGE ERROR: {error}")
        self.errors.append({"timestamp": timestamp, "error": str(error)})
        
    async def capture_training_status(self):
        """Capture current training status elements"""
        try:
            # Get status indicator
            status_element = await self.page.query_selector('[class*="status"], [data-testid="training-status"]')
            status_text = await status_element.text_content() if status_element else "Not found"
            
            # Get progress indicator
            progress_element = await self.page.query_selector('[class*="progress"], [data-testid="progress"]')
            progress_text = await progress_element.text_content() if progress_element else "Not found"
            
            # Get step count
            step_element = await self.page.query_selector('[class*="step"], [data-testid="current-step"]')
            step_text = await step_element.text_content() if step_element else "Not found"
            
            # Check for chart presence
            chart_canvas = await self.page.query_selector('canvas')
            chart_present = chart_canvas is not None
            
            # Check for "No training data available" message
            no_data_msg = await self.page.query_selector('text="No training data available"')
            no_data_present = no_data_msg is not None
            
            return {
                "timestamp": datetime.now().isoformat(),
                "status": status_text,
                "progress": progress_text,
                "steps": step_text,
                "chart_present": chart_present,
                "no_data_message": no_data_present
            }
        except Exception as e:
            print(f"⚠️  Error capturing status: {e}")
            return None
            
    async def start_training_and_monitor(self):
        """Start training and monitor the process"""
        print("🎯 Starting training session...")
        
        # Look for Start Training button and click it
        try:
            start_button = await self.page.wait_for_selector(
                'button:has-text("Start Training"), [data-testid="start-training"]',
                timeout=5000
            )
            await start_button.click()
            print("✅ Training started!")
        except Exception as e:
            print(f"❌ Could not find/click start training button: {e}")
            return
            
        # Monitor for 60 seconds or until training completes
        print("👀 Monitoring training progress...")
        start_time = time.time()
        last_chart_check = 0
        
        while time.time() - start_time < 60:  # Monitor for 60 seconds
            # Capture current status
            status = await self.capture_training_status()
            if status:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"📊 [{timestamp}] Status: {status['status']} | Progress: {status['progress']} | Steps: {status['steps']} | Chart: {'✅' if status['chart_present'] else '❌'} | NoData: {'✅' if status['no_data_message'] else '❌'}")
                self.status_updates.append(status)
                
                # Check for chart data every 5 seconds
                if time.time() - last_chart_check > 5:
                    await self.check_chart_data()
                    last_chart_check = time.time()
                    
            await asyncio.sleep(2)  # Check every 2 seconds
            
        print("⏰ Monitoring period complete")
        
    async def check_chart_data(self):
        """Check if chart has data and count data points"""
        try:
            # Execute JavaScript to get chart data
            chart_data = await self.page.evaluate("""
                () => {
                    // Look for Chart.js instance
                    const canvas = document.querySelector('canvas');
                    if (!canvas) return { error: 'No canvas found' };
                    
                    // Try to get Chart.js instance
                    const chartInstance = Chart.getChart(canvas);
                    if (!chartInstance) return { error: 'No Chart.js instance found' };
                    
                    const datasets = chartInstance.data.datasets;
                    if (!datasets || datasets.length === 0) return { dataPoints: 0, datasets: [] };
                    
                    let totalPoints = 0;
                    const datasetInfo = [];
                    
                    datasets.forEach((dataset, i) => {
                        const points = dataset.data ? dataset.data.length : 0;
                        totalPoints += points;
                        datasetInfo.push({
                            label: dataset.label,
                            points: points,
                            lastPoint: points > 0 ? dataset.data[points - 1] : null
                        });
                    });
                    
                    return { dataPoints: totalPoints, datasets: datasetInfo };
                }
            """)
            
            if 'error' not in chart_data:
                timestamp = datetime.now().strftime("%H:%M:%S")
                total_points = chart_data['dataPoints']
                datasets_info = chart_data['datasets']
                
                print(f"📈 [{timestamp}] Chart Data: {total_points} total points")
                for dataset in datasets_info:
                    last_point_str = f" (last: {dataset['lastPoint']})" if dataset['lastPoint'] else ""
                    print(f"   📊 {dataset['label']}: {dataset['points']} points{last_point_str}")
                
                self.chart_data_points.append({
                    "timestamp": timestamp,
                    "total_points": total_points,
                    "datasets": datasets_info
                })
            else:
                print(f"⚠️  Chart check error: {chart_data['error']}")
                
        except Exception as e:
            print(f"❌ Error checking chart data: {e}")
            
    async def generate_report(self):
        """Generate monitoring report"""
        print("\n" + "="*60)
        print("📋 TRAINING MONITORING REPORT")
        print("="*60)
        
        print(f"\n🕐 Monitoring Duration: {len(self.status_updates)} status checks")
        print(f"❌ Errors Detected: {len(self.errors)}")
        print(f"📈 Chart Data Snapshots: {len(self.chart_data_points)}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   [{error['timestamp']}] {error['error']}")
                
        if self.chart_data_points:
            print("\n📈 CHART DATA PROGRESSION:")
            for point in self.chart_data_points:
                print(f"   [{point['timestamp']}] {point['total_points']} total points")
                
        # Check for issues
        issues = []
        
        # Check if chart never appeared
        chart_appeared = any(status.get('chart_present', False) for status in self.status_updates)
        if not chart_appeared:
            issues.append("🚨 Chart never appeared during monitoring")
            
        # Check for inconsistent step counts
        step_texts = [status.get('steps', '') for status in self.status_updates if status.get('steps')]
        if len(set(step_texts)) > 1:
            issues.append(f"⚠️  Inconsistent step counts detected: {set(step_texts)}")
            
        # Check for chart data timing
        if self.chart_data_points:
            first_data = self.chart_data_points[0]
            last_data = self.chart_data_points[-1]
            if first_data['total_points'] == last_data['total_points'] == 0:
                issues.append("🚨 Chart never received data points")
                
        if issues:
            print("\n🚨 ISSUES DETECTED:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ No major issues detected")
            
        print("\n" + "="*60)
        
    async def cleanup(self):
        """Cleanup browser resources"""
        if self.browser:
            await self.browser.close()
            
async def main():
    monitor = TrainingMonitor()
    
    try:
        await monitor.setup()
        await monitor.start_training_and_monitor()
        await monitor.generate_report()
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted by user")
    except Exception as e:
        print(f"❌ Monitoring error: {e}")
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())