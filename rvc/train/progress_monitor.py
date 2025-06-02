import json
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any


class TrainingProgressMonitor:
    """Monitors and reports training progress for batch training"""
    
    def __init__(self, model_name: str, total_epochs: int):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.progress_file = os.path.join("logs", model_name, "training_progress.json")
        self.start_time = datetime.now()
        self.epoch_start_time = None
        self.first_epoch_duration = None
        
        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(self.progress_file), exist_ok=True)
        
        # Initialize progress file
        self._write_progress({
            "model_name": model_name,
            "total_epochs": total_epochs,
            "current_epoch": 0,
            "status": "starting",
            "start_time": self.start_time.isoformat(),
            "progress_percentage": 0.0,
            "estimated_time_remaining": None,
            "time_per_epoch": None,
            "current_step": "Initializing training",
            "last_updated": datetime.now().isoformat()
        })
    
    def start_epoch(self, epoch: int):
        """Called at the start of each epoch"""
        self.epoch_start_time = datetime.now()
        
        progress_data = {
            "current_epoch": epoch,
            "status": "training",
            "progress_percentage": (epoch - 1) / self.total_epochs * 100,
            "current_step": f"Training epoch {epoch}/{self.total_epochs}",
            "last_updated": datetime.now().isoformat()
        }
        
        # Calculate time estimates after first epoch
        if epoch > 1 and self.first_epoch_duration:
            remaining_epochs = self.total_epochs - epoch + 1
            estimated_remaining = remaining_epochs * self.first_epoch_duration
            progress_data["estimated_time_remaining"] = estimated_remaining
            progress_data["time_per_epoch"] = self.first_epoch_duration
        
        self._write_progress(progress_data)
    
    def end_epoch(self, epoch: int):
        """Called at the end of each epoch"""
        if self.epoch_start_time:
            epoch_duration = (datetime.now() - self.epoch_start_time).total_seconds()
            
            # Store first epoch duration for time estimation
            if epoch == 1:
                self.first_epoch_duration = epoch_duration
            
            progress_data = {
                "current_epoch": epoch,
                "status": "training",
                "progress_percentage": epoch / self.total_epochs * 100,
                "current_step": f"Completed epoch {epoch}/{self.total_epochs}",
                "time_per_epoch": epoch_duration if epoch == 1 else self.first_epoch_duration,
                "last_updated": datetime.now().isoformat()
            }
            
            # Calculate remaining time estimate
            if self.first_epoch_duration and epoch >= 1:
                remaining_epochs = self.total_epochs - epoch
                estimated_remaining = remaining_epochs * self.first_epoch_duration
                progress_data["estimated_time_remaining"] = estimated_remaining
            
            self._write_progress(progress_data)
    
    def update_step(self, step_description: str):
        """Update the current step description"""
        self._write_progress({
            "current_step": step_description,
            "last_updated": datetime.now().isoformat()
        })
    
    def set_status(self, status: str):
        """Update the training status"""
        self._write_progress({
            "status": status,
            "last_updated": datetime.now().isoformat()
        })
    
    def complete_training(self):
        """Called when training is completed"""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        self._write_progress({
            "status": "completed",
            "progress_percentage": 100.0,
            "current_step": "Training completed",
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "last_updated": datetime.now().isoformat()
        })
    
    def set_error(self, error_message: str):
        """Called when training encounters an error"""
        self._write_progress({
            "status": "error",
            "error_message": error_message,
            "last_updated": datetime.now().isoformat()
        })
    
    def _write_progress(self, data: Dict[str, Any]):
        """Write progress data to file"""
        try:
            # Read existing data
            existing_data = {}
            if os.path.exists(self.progress_file):
                try:
                    with open(self.progress_file, 'r') as f:
                        existing_data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
            
            # Update with new data
            existing_data.update(data)
            
            # Write back to file
            with open(self.progress_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"Error writing progress file: {e}")
    
    @classmethod
    def read_progress(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Read progress data from file"""
        progress_file = os.path.join("logs", model_name, "training_progress.json")
        
        if not os.path.exists(progress_file):
            return None
        
        try:
            with open(progress_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
    
    @classmethod
    def cleanup_progress_file(cls, model_name: str):
        """Remove progress file after training"""
        progress_file = os.path.join("logs", model_name, "training_progress.json")
        if os.path.exists(progress_file):
            try:
                os.remove(progress_file)
            except OSError:
                pass
