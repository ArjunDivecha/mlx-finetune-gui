"""
MLX Fine-Tuning GUI Backend API Server

This FastAPI server provides REST endpoints and WebSocket connections
for the MLX fine-tuning GUI application.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import os
import sys
import subprocess
import signal
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to path to import existing modules
sys.path.append('/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune')

app = FastAPI(title="MLX Fine-Tuning GUI API", version="1.0.0")

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass
class TrainingConfig:
    """Training configuration data class"""
    model_path: str
    train_data_path: str
    val_data_path: str
    learning_rate: float = 1e-5
    batch_size: int = 1
    max_seq_length: int = 1024
    iterations: int = 7329
    steps_per_report: int = 25
    steps_per_eval: int = 200
    save_every: int = 1000
    early_stop: bool = True
    patience: int = 3
    adapter_name: str = "mlx_finetune"

class TrainingManager:
    """Manages training processes and state"""
    
    def __init__(self):
        self.current_process: Optional[subprocess.Popen] = None
        self.training_state = "idle"  # idle, running, paused, completed, error
        self.current_config: Optional[TrainingConfig] = None
        self.training_metrics: Dict[str, Any] = {}
        self.websocket_clients: List[WebSocket] = []
        self.output_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters"
        self.log_file = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/logs/gui_training.log"
        self.sessions_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/sessions"
        self.current_session_id: Optional[str] = None
        
        # Ensure sessions directory exists
        os.makedirs(self.sessions_dir, exist_ok=True)
        
        # Load the most recent session on startup
        self.load_latest_session()
    
    def save_session(self):
        """Save current training session to persistent storage"""
        if not self.current_config or not self.training_metrics:
            return
            
        try:
            # Generate session ID if not already set
            if not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
            
            session_data = {
                "session_id": self.current_session_id,
                "timestamp": datetime.now().isoformat(),
                "training_state": self.training_state,
                "config": asdict(self.current_config),
                "metrics": self.training_metrics,
                "adapter_path": f"{self.output_dir}/{self.current_config.adapter_name}"
            }
            
            session_file = os.path.join(self.sessions_dir, f"session_{self.current_session_id}.json")
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            # Update latest session pointer
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            with open(latest_file, 'w') as f:
                json.dump({"latest_session_id": self.current_session_id}, f)
                
            logger.info(f"Saved training session: {self.current_session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """Load a specific training session"""
        try:
            session_file = os.path.join(self.sessions_dir, f"session_{session_id}.json")
            if not os.path.exists(session_file):
                logger.warning(f"Session file not found: {session_file}")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Restore training configuration
            config_data = session_data["config"]
            self.current_config = TrainingConfig(**config_data)
            
            # Restore metrics and state
            self.training_metrics = session_data["metrics"]
            self.training_state = session_data["training_state"]
            self.current_session_id = session_data["session_id"]
            
            logger.info(f"Loaded training session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def load_latest_session(self):
        """Load the most recent training session on startup"""
        try:
            latest_file = os.path.join(self.sessions_dir, "latest.json")
            if not os.path.exists(latest_file):
                logger.info("No previous training sessions found")
                return
            
            with open(latest_file, 'r') as f:
                latest_data = json.load(f)
            
            session_id = latest_data.get("latest_session_id")
            if session_id:
                success = self.load_session(session_id)
                if success:
                    logger.info(f"Restored previous training session: {session_id}")
                    # Only restore if training was completed
                    if self.training_state not in ["completed", "error", "stopped"]:
                        self.training_state = "idle"
                        logger.info("Previous session was not completed, reset to idle state")
                else:
                    logger.warning("Failed to restore previous session")
            
        except Exception as e:
            logger.error(f"Failed to load latest session: {e}")
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all saved training sessions"""
        sessions = []
        try:
            if not os.path.exists(self.sessions_dir):
                return sessions
                
            for filename in os.listdir(self.sessions_dir):
                if filename.startswith("session_") and filename.endswith(".json"):
                    session_file = os.path.join(self.sessions_dir, filename)
                    try:
                        with open(session_file, 'r') as f:
                            session_data = json.load(f)
                        
                        # Create summary for session list
                        session_summary = {
                            "session_id": session_data["session_id"],
                            "timestamp": session_data["timestamp"],
                            "training_state": session_data["training_state"],
                            "model_name": session_data["config"]["model_path"].split('/')[-1],
                            "adapter_name": session_data["config"]["adapter_name"],
                            "final_train_loss": session_data["metrics"].get("train_loss", 0),
                            "final_val_loss": session_data["metrics"].get("val_loss", 0),
                            "steps_completed": session_data["metrics"].get("current_step", 0),
                            "total_steps": session_data["metrics"].get("total_steps", 0)
                        }
                        sessions.append(session_summary)
                        
                    except Exception as e:
                        logger.error(f"Error reading session file {filename}: {e}")
                        continue
            
            # Sort by timestamp, newest first
            sessions.sort(key=lambda x: x["timestamp"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get sessions list: {e}")
        
        return sessions
        
    async def add_websocket(self, websocket: WebSocket):
        """Add a WebSocket client"""
        self.websocket_clients.append(websocket)
        # Send current state
        await websocket.send_json({
            "type": "training_state",
            "data": {
                "state": self.training_state,
                "metrics": self.training_metrics
            }
        })
        
    def remove_websocket(self, websocket: WebSocket):
        """Remove a WebSocket client"""
        if websocket in self.websocket_clients:
            self.websocket_clients.remove(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket clients"""
        if self.websocket_clients:
            disconnected = []
            for client in self.websocket_clients:
                try:
                    await client.send_json(message)
                except:
                    disconnected.append(client)
            
            # Remove disconnected clients
            for client in disconnected:
                self.remove_websocket(client)
    
    async def start_training(self, config: TrainingConfig) -> bool:
        """Start training with the given configuration"""
        if self.current_process and self.current_process.poll() is None:
            raise HTTPException(status_code=400, detail="Training is already running")
        
        self.current_config = config
        self.training_state = "running"
        self.training_metrics = {
            "current_step": 0,
            "total_steps": config.iterations,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "learning_rate": config.learning_rate,
            "start_time": datetime.now().isoformat(),
            "estimated_time_remaining": None
        }
        
        # Generate new session ID for this training run
        self.current_session_id = str(uuid.uuid4())
        
        # Create config file for the training script
        config_data = {
            "venv_python": "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python",
            "base_model_dir": config.model_path,
            "chat_train_jsonl": config.train_data_path,
            "chat_valid_jsonl": config.val_data_path,
            "prepare_from_chat": True,
            "apply_chat_template_model_dir": config.model_path,
            "out_data_dir": "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune/data",
            "adapter_output_dir": self.output_dir,
            "adapter_name": config.adapter_name,
            "optimizer": "adamw",
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "iters": config.iterations,
            "steps_per_report": config.steps_per_report,
            "steps_per_eval": config.steps_per_eval,
            "val_batches": -1,
            "max_seq_length": config.max_seq_length,
            "grad_checkpoint": True,
            "mask_prompt": False,
            "save_every": config.save_every,
            "resume_adapter_file": None,
            "train_log": self.log_file,
            "enable_early_stop": config.early_stop,
            "no_improve_patience_evals": config.patience
        }
        
        # Write config file
        config_path = "/tmp/gui_training_config.yaml"
        with open(config_path, 'w') as f:
            import yaml
            yaml.dump(config_data, f, default_flow_style=False)
        
        # Start training process
        try:
            cmd = [
                "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python",
                "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/one_step_finetune/run_finetune.py",
                "--config", config_path
            ]
            
            self.current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid
            )
            
            # Start monitoring the process
            asyncio.create_task(self._monitor_training())
            
            await self.broadcast({
                "type": "training_started",
                "data": {"config": config_data}
            })
            
            return True
            
        except Exception as e:
            self.training_state = "error"
            logger.error(f"Failed to start training: {e}")
            await self.broadcast({
                "type": "training_error",
                "data": {"error": str(e)}
            })
            return False
    
    async def stop_training(self):
        """Stop the current training process"""
        if self.current_process and self.current_process.poll() is None:
            try:
                # Send SIGTERM to the process group
                os.killpg(self.current_process.pid, signal.SIGTERM)
                
                # Wait for graceful shutdown
                try:
                    self.current_process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    os.killpg(self.current_process.pid, signal.SIGKILL)
                
                self.training_state = "stopped"
                # Save stopped session
                self.save_session()
                await self.broadcast({
                    "type": "training_stopped",
                    "data": {}
                })
                
            except Exception as e:
                logger.error(f"Error stopping training: {e}")
        
    async def _monitor_training(self):
        """Monitor the training process and broadcast updates"""
        if not self.current_process:
            return
            
        try:
            import re
            
            # Patterns to extract metrics from training output
            step_pattern = re.compile(r'Iter (\d+):')
            loss_pattern = re.compile(r'Train loss ([0-9.]+)')
            val_pattern = re.compile(r'Val loss\s+([0-9.]+)')
            lr_pattern = re.compile(r'Learning Rate ([0-9.e-]+)')
            
            while self.current_process and self.current_process.poll() is None:
                try:
                    output = self.current_process.stdout.readline()
                    if output:
                        # Parse metrics from output
                        step_match = step_pattern.search(output)
                        if step_match:
                            # Convert 0-based iteration to 1-based step count
                            self.training_metrics["current_step"] = int(step_match.group(1)) + 1
                        
                        loss_match = loss_pattern.search(output)
                        if loss_match:
                            self.training_metrics["train_loss"] = float(loss_match.group(1))
                        
                        val_match = val_pattern.search(output)
                        if val_match:
                            self.training_metrics["val_loss"] = float(val_match.group(1))
                        
                        lr_match = lr_pattern.search(output)
                        if lr_match:
                            self.training_metrics["learning_rate"] = float(lr_match.group(1))
                        
                        # Calculate progress and ETA
                        if "current_step" in self.training_metrics and "total_steps" in self.training_metrics:
                            progress = self.training_metrics["current_step"] / self.training_metrics["total_steps"]
                            if progress > 0 and "start_time" in self.training_metrics:
                                start_time = datetime.fromisoformat(self.training_metrics["start_time"])
                                elapsed = (datetime.now() - start_time).total_seconds()
                                estimated_total = elapsed / progress
                                remaining = estimated_total - elapsed
                                self.training_metrics["estimated_time_remaining"] = remaining
                        
                        # Broadcast update
                        await self.broadcast({
                            "type": "training_progress",
                            "data": {
                                "metrics": self.training_metrics,
                                "log_line": output.strip()
                            }
                        })
                
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error monitoring training: {e}")
                    break
            
            # Process completed - but read any remaining output first
            return_code = self.current_process.wait()
            
            # Read any remaining output after process completion
            try:
                while True:
                    remaining_output = self.current_process.stdout.readline()
                    if not remaining_output:
                        break
                    
                    # Parse final metrics from remaining output
                    step_match = step_pattern.search(remaining_output)
                    if step_match:
                        # Convert 0-based iteration to 1-based step count
                        self.training_metrics["current_step"] = int(step_match.group(1)) + 1
                    
                    loss_match = loss_pattern.search(remaining_output)
                    if loss_match:
                        self.training_metrics["train_loss"] = float(loss_match.group(1))
                    
                    val_match = val_pattern.search(remaining_output)
                    if val_match:
                        self.training_metrics["val_loss"] = float(val_match.group(1))
            except:
                pass  # Ignore errors when reading final output
            
            if return_code == 0:
                self.training_state = "completed"
                # Save completed session
                self.save_session()
                await self.broadcast({
                    "type": "training_completed",
                    "data": {"final_metrics": self.training_metrics}
                })
            else:
                self.training_state = "error"
                # Save error session
                self.save_session()
                await self.broadcast({
                    "type": "training_error",
                    "data": {"error": f"Training process exited with code {return_code}"}
                })
                
        except Exception as e:
            self.training_state = "error"
            logger.error(f"Training monitoring error: {e}")
            await self.broadcast({
                "type": "training_error",
                "data": {"error": str(e)}
            })

# Global training manager instance
training_manager = TrainingManager()

# REST API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    """List available models"""
    models_dir = "/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/base_model"
    models = []
    
    try:
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                if os.path.isdir(item_path):
                    # Check if it's a valid model directory (contains config.json)
                    config_path = os.path.join(item_path, "config.json")
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                models.append({
                                    "name": item,
                                    "path": item_path,
                                    "model_type": config.get("model_type", "unknown"),
                                    "vocab_size": config.get("vocab_size", 0)
                                })
                        except Exception as e:
                            logger.warning(f"Could not read config for {item}: {e}")
                            models.append({
                                "name": item,
                                "path": item_path,
                                "model_type": "unknown",
                                "vocab_size": 0
                            })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
    
    return {"models": models}

@app.get("/training/status")
async def get_training_status():
    """Get current training status"""
    return {
        "state": training_manager.training_state,
        "metrics": training_manager.training_metrics,
        "config": training_manager.current_config.__dict__ if training_manager.current_config else None
    }

@app.post("/training/start")
async def start_training(config_data: Dict[str, Any], background_tasks: BackgroundTasks):
    """Start training with given configuration"""
    try:
        config = TrainingConfig(
            model_path=config_data["model_path"],
            train_data_path=config_data["train_data_path"],
            val_data_path=config_data.get("val_data_path", ""),
            learning_rate=config_data.get("learning_rate", 1e-5),
            batch_size=config_data.get("batch_size", 1),
            max_seq_length=config_data.get("max_seq_length", 1024),
            iterations=config_data.get("iterations", 7329),
            steps_per_report=config_data.get("steps_per_report", 25),
            steps_per_eval=config_data.get("steps_per_eval", 200),
            save_every=config_data.get("save_every", 1000),
            early_stop=config_data.get("early_stop", True),
            patience=config_data.get("patience", 3),
            adapter_name=config_data.get("adapter_name", "mlx_finetune")
        )
        
        success = await training_manager.start_training(config)
        if success:
            return {"status": "started", "message": "Training started successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start training")
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/training/stop")
async def stop_training():
    """Stop current training"""
    await training_manager.stop_training()
    return {"status": "stopped", "message": "Training stop requested"}

@app.get("/training/logs")
async def get_training_logs():
    """Get training logs"""
    try:
        if os.path.exists(training_manager.log_file):
            with open(training_manager.log_file, 'r') as f:
                logs = f.readlines()[-100:]  # Last 100 lines
            return {"logs": logs}
        else:
            return {"logs": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_sessions():
    """Get list of all training sessions"""
    sessions = training_manager.get_all_sessions()
    return {"sessions": sessions}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get details of a specific training session"""
    try:
        session_file = os.path.join(training_manager.sessions_dir, f"session_{session_id}.json")
        if not os.path.exists(session_file):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return session_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/load")
async def load_session(session_id: str):
    """Load a specific training session"""
    try:
        success = training_manager.load_session(session_id)
        if success:
            # Broadcast the loaded session state to any connected clients
            await training_manager.broadcast({
                "type": "session_loaded",
                "data": {
                    "session_id": session_id,
                    "state": training_manager.training_state,
                    "metrics": training_manager.training_metrics,
                    "config": asdict(training_manager.current_config) if training_manager.current_config else None
                }
            })
            return {"status": "success", "message": f"Session {session_id} loaded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found or could not be loaded")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/test-base")
async def test_base_model(request_data: dict):
    """Test the base model (without adapter) with a prompt"""
    try:
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 100)
        temperature = request_data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Get the model path from the latest training config
        if not training_manager.current_config:
            raise HTTPException(status_code=400, detail="No training session found. Please complete a training session first.")
        
        config = training_manager.current_config
        model_path = config.model_path
        
        # Use MLX to generate text with the base model only (no adapter)
        python_path = '/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python'
        
        # Create a simple inference command using mlx-lm for base model only
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

# Load the base model WITHOUT adapter
model, tokenizer = load("{model_path}")

# Generate text
prompt = """{prompt}"""

# Try different parameter combinations based on MLX version
response = None
try:
    # Try with temp parameter
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens}, temp={temperature})
except TypeError:
    try:
        # Try with temperature parameter  
        response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens}, temperature={temperature})
    except TypeError:
        try:
            # Try with just basic parameters
            response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
        except Exception as e:
            print("RESPONSE_START")
            print(f"Error: Could not generate response - {{str(e)}}")
            print("RESPONSE_END")
            exit(1)

print("RESPONSE_START")
print(response)
print("RESPONSE_END")
'''
        ]
        
        # Run the inference
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if process.returncode != 0:
            logger.error(f"Base model test failed: {process.stderr}")
            raise HTTPException(status_code=500, detail=f"Base model inference failed: {process.stderr}")
        
        # Parse the response
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            response_start = output.find("RESPONSE_START") + len("RESPONSE_START\n")
            response_end = output.find("RESPONSE_END")
            generated_text = output[response_start:response_end].strip()
        else:
            generated_text = output.strip()
        
        return {
            "success": True,
            "prompt": prompt,
            "response": generated_text,
            "model_info": {
                "base_model": model_path.split('/')[-1],
                "adapter": "none (base model)",
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Base model inference timed out")
    except Exception as e:
        logger.error(f"Base model test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/test")
async def test_model(request_data: dict):
    """Test the fine-tuned model with a prompt"""
    try:
        prompt = request_data.get("prompt", "")
        max_tokens = request_data.get("max_tokens", 100)
        temperature = request_data.get("temperature", 0.7)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Get the model and adapter paths from the latest training config
        if not training_manager.current_config:
            raise HTTPException(status_code=400, detail="No training session found. Please complete a training session first.")
        
        config = training_manager.current_config
        model_path = config.model_path
        adapter_name = config.adapter_name
        
        # Path to the fine-tuned adapter
        adapter_path = f"/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/artifacts/lora_adapters/{adapter_name}"
        
        # Check if adapter exists
        if not os.path.exists(adapter_path):
            raise HTTPException(status_code=404, detail=f"Fine-tuned adapter not found at {adapter_path}")
        
        # Use MLX to generate text with the fine-tuned model
        # This is a simplified implementation - you might want to use a proper MLX inference script
        python_path = '/Users/macbook2024/Dropbox/AAA Backup/A Working/Arjun LLM Writing/local_qwen/.venv/bin/python'
        
        # Create a simple inference command using mlx-lm
        cmd = [
            python_path, '-c', f'''
import mlx.core as mx
from mlx_lm import load, generate

# Load the base model and adapter
model, tokenizer = load("{model_path}", adapter_path="{adapter_path}")

# Generate text
prompt = """{prompt}"""

# Try different parameter combinations based on MLX version
response = None
try:
    # Try with temp parameter
    response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens}, temp={temperature})
except TypeError:
    try:
        # Try with temperature parameter  
        response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens}, temperature={temperature})
    except TypeError:
        try:
            # Try with just basic parameters
            response = generate(model, tokenizer, prompt=prompt, max_tokens={max_tokens})
        except Exception as e:
            print("RESPONSE_START")
            print(f"Error: Could not generate response - {{str(e)}}")
            print("RESPONSE_END")
            exit(1)

print("RESPONSE_START")
print(response)
print("RESPONSE_END")
'''
        ]
        
        # Run the inference
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if process.returncode != 0:
            logger.error(f"Model test failed: {process.stderr}")
            raise HTTPException(status_code=500, detail=f"Model inference failed: {process.stderr}")
        
        # Parse the response
        output = process.stdout
        if "RESPONSE_START" in output and "RESPONSE_END" in output:
            response_start = output.find("RESPONSE_START") + len("RESPONSE_START\n")
            response_end = output.find("RESPONSE_END")
            generated_text = output[response_start:response_end].strip()
        else:
            generated_text = output.strip()
        
        return {
            "success": True,
            "prompt": prompt,
            "response": generated_text,
            "model_info": {
                "base_model": model_path.split('/')[-1],
                "adapter": adapter_name,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Model inference timed out")
    except Exception as e:
        logger.error(f"Model test error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training updates"""
    await websocket.accept()
    await training_manager.add_websocket(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        training_manager.remove_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)