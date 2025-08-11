#!/usr/bin/env python3
"""
Simple Training Monitor
Monitors the backend API and training logs directly
"""

import requests
import time
import json
import os
from datetime import datetime

class SimpleMonitor:
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.log_file = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/logs/gui_training.log"
        self.last_log_position = 0
        self.training_updates = []
        self.chart_data_issues = []
        
    def get_training_status(self):
        """Get current training status from backend"""
        try:
            response = requests.get(f"{self.backend_url}/training/status", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
            
    def read_new_logs(self):
        """Read new lines from training log"""
        if not os.path.exists(self.log_file):
            return []
            
        try:
            with open(self.log_file, 'r') as f:
                f.seek(self.last_log_position)
                new_lines = f.readlines()
                self.last_log_position = f.tell()
                return [line.strip() for line in new_lines if line.strip()]
        except Exception as e:
            print(f"❌ Error reading log: {e}")
            return []
            
    def analyze_training_data(self, status):
        """Analyze training status for inconsistencies"""
        issues = []
        
        if 'metrics' in status and status['metrics']:
            metrics = status['metrics']
            current_step = metrics.get('current_step', 0)
            total_steps = metrics.get('total_steps', 0)
            
            # Check if steps make sense
            if current_step > total_steps and total_steps > 0:
                issues.append(f"🚨 Current step ({current_step}) > Total steps ({total_steps})")
                
            # Check if training is running but no progress
            if status.get('state') == 'running' and current_step == 0:
                issues.append("⚠️  Training state is 'running' but current_step is 0")
                
            # Check for strange progress percentages
            if total_steps > 0:
                progress_pct = (current_step / total_steps) * 100
                if progress_pct > 100:
                    issues.append(f"🚨 Progress > 100%: {progress_pct:.1f}% ({current_step}/{total_steps})")
                    
        return issues
        
    def monitor(self, duration_seconds=60):
        """Monitor training for specified duration"""
        print(f"🚀 Starting {duration_seconds}s monitoring session...")
        print(f"📡 Backend: {self.backend_url}")
        print(f"📄 Log file: {self.log_file}")
        print("-" * 60)
        
        start_time = time.time()
        last_status_check = 0
        last_log_check = 0
        
        while time.time() - start_time < duration_seconds:
            current_time = time.time()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Check status every 3 seconds
            if current_time - last_status_check >= 3:
                status = self.get_training_status()
                
                if 'error' in status:
                    print(f"❌ [{timestamp}] Backend error: {status['error']}")
                else:
                    state = status.get('state', 'unknown')
                    metrics = status.get('metrics', {})
                    
                    current_step = metrics.get('current_step', 0)
                    total_steps = metrics.get('total_steps', 0)
                    train_loss = metrics.get('train_loss', 0)
                    
                    progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
                    
                    print(f"📊 [{timestamp}] State: {state} | Step: {current_step}/{total_steps} ({progress_pct:.1f}%) | Loss: {train_loss:.4f}")
                    
                    # Analyze for issues
                    issues = self.analyze_training_data(status)
                    for issue in issues:
                        print(f"   {issue}")
                        self.chart_data_issues.append({"timestamp": timestamp, "issue": issue})
                    
                    self.training_updates.append({
                        "timestamp": timestamp,
                        "status": status,
                        "issues": issues
                    })
                    
                last_status_check = current_time
                
            # Check logs every 2 seconds
            if current_time - last_log_check >= 2:
                new_logs = self.read_new_logs()
                for log_line in new_logs:
                    if any(keyword in log_line.lower() for keyword in ['iter', 'loss', 'train', 'val']):
                        print(f"📝 [{timestamp}] LOG: {log_line}")
                last_log_check = current_time
                
            time.sleep(1)
            
        self.generate_report()
        
    def generate_report(self):
        """Generate final monitoring report"""
        print("\n" + "="*60)
        print("📋 TRAINING MONITORING REPORT")
        print("="*60)
        
        print(f"\n📊 Total Status Updates: {len(self.training_updates)}")
        print(f"🚨 Issues Detected: {len(self.chart_data_issues)}")
        
        if self.chart_data_issues:
            print("\n🚨 ISSUES FOUND:")
            for issue_record in self.chart_data_issues:
                print(f"   [{issue_record['timestamp']}] {issue_record['issue']}")
                
        # Analyze training progression
        if len(self.training_updates) > 1:
            first_update = self.training_updates[0]
            last_update = self.training_updates[-1]
            
            first_step = first_update['status'].get('metrics', {}).get('current_step', 0)
            last_step = last_update['status'].get('metrics', {}).get('current_step', 0)
            
            print(f"\n📈 TRAINING PROGRESSION:")
            print(f"   Start step: {first_step}")
            print(f"   End step: {last_step}")
            print(f"   Progress: {last_step - first_step} steps")
            
            # Check for stalled training
            if last_step == first_step and first_update['status'].get('state') == 'running':
                print("   ⚠️  WARNING: Training appears stalled (no step progress)")
                
        # Summary of states seen
        states_seen = set()
        for update in self.training_updates:
            states_seen.add(update['status'].get('state', 'unknown'))
        print(f"\n🔄 States observed: {', '.join(sorted(states_seen))}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    monitor = SimpleMonitor()
    try:
        monitor.monitor(60)  # Monitor for 60 seconds
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring interrupted by user")
        monitor.generate_report()
    except Exception as e:
        print(f"❌ Monitoring error: {e}")