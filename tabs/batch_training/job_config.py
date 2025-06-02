import os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import uuid


@dataclass
class BatchTrainingJob:
    """Configuration for a single training job in a batch"""
    
    # Unique identifier and metadata
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_name: str = "Training Job - "
    
    # Model Settings
    model_name: str = "my-project"
    architecture: str = "RVC"
    sampling_rate: str = "48000"
    vocoder: str = "HiFi-GAN"
    cpu_cores: int = 4
    gpu: str = "0"
    
    # Dataset and Preprocessing
    dataset_path: str = ""
    cut_preprocess: str = "Automatic"
    process_effects: bool = True
    noise_reduction: bool = False
    clean_strength: float = 0.5
    chunk_len: float = 3.0
    overlap_len: float = 0.3
    
    # Extract Settings
    f0_method: str = "rmvpe"
    embedder_model: str = "contentvec"
    embedder_model_custom: Optional[str] = None
    include_mutes: int = 2
    hop_length: int = 128
    
    # Training Settings
    batch_size: int = 8
    save_every_epoch: int = 10
    total_epoch: int = 500
    save_only_latest: bool = True
    save_every_weights: bool = True
    pretrained: bool = True
    cleanup: bool = False
    cache_dataset_in_gpu: bool = False
    checkpointing: bool = False
    
    # Advanced Training Settings
    custom_pretrained: bool = False
    g_pretrained_path: Optional[str] = None
    d_pretrained_path: Optional[str] = None
    overtraining_detector: bool = False
    overtraining_threshold: int = 50
    index_algorithm: str = "Auto"
    
    # Job Status
    status: str = "pending"  # pending, preprocessing, extracting, training, indexing, completed, failed
    current_step: str = ""
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Training Progress Tracking
    current_epoch: int = 0
    first_epoch_start_time: Optional[datetime] = None
    first_epoch_end_time: Optional[datetime] = None
    estimated_total_time: Optional[float] = None  # in seconds
    time_per_epoch: Optional[float] = None  # in seconds
    created_on: datetime = datetime.now()
    
    def __post_init__(self):
        """Set default job name if not provided"""
        if not self.job_name:
            self.job_name = f"Training Job - {self.model_name}"
    
    def to_dict(self) -> dict:
        """Convert job to dictionary for serialization"""
        data = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            else:
                data[key] = value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BatchTrainingJob':
        """Create job from dictionary"""
        # Handle datetime fields
        for field_name in ['start_time', 'end_time', 'first_epoch_start_time', 'first_epoch_end_time']:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)
    
    def get_status_display(self) -> str:
        """Get human-readable status"""
        status_map = {
            "pending": "⏳ Pending",
            "preprocessing": "🔄 Preprocessing",
            "extracting": "🔍 Extracting Features",
            "training": "🎯 Training",
            "indexing": "📊 Generating Index",
            "completed": "✅ Completed",
            "failed": "❌ Failed"
        }
        return status_map.get(self.status, self.status)
    
    def get_duration_display(self) -> str:
        """Get job duration if completed"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return "N/A"
    
    def calculate_time_estimates(self) -> None:
        """Calculate time estimates after first epoch completion"""
        if self.first_epoch_start_time and self.first_epoch_end_time:
            # Calculate time per epoch
            epoch_duration = (self.first_epoch_end_time - self.first_epoch_start_time).total_seconds()
            self.time_per_epoch = epoch_duration
            
            # Calculate estimated total time (remaining epochs * time per epoch)
            remaining_epochs = self.total_epoch - self.current_epoch
            self.estimated_total_time = remaining_epochs * epoch_duration
    
    def get_epoch_display(self) -> str:
        """Get current epoch display"""
        if self.status == "training" and self.current_epoch > 0:
            return f"Epoch {self.current_epoch}/{self.total_epoch}"
        return ""
    
    def get_time_estimate_display(self) -> str:
        """Get estimated remaining time display"""
        if self.estimated_total_time is not None and self.status == "training":
            hours, remainder = divmod(self.estimated_total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"Est. remaining: {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"
        return ""
    
    def get_time_per_epoch_display(self) -> str:
        """Get time per epoch display"""
        if self.time_per_epoch is not None:
            minutes, seconds = divmod(self.time_per_epoch, 60)
            return f"Time/epoch: {int(minutes):02d}m {int(seconds):02d}s"
        return ""
    
    def get_detailed_progress_display(self) -> str:
        """Get detailed progress display including epoch and time info"""
        parts = []
        
        # Add epoch info if training
        epoch_info = self.get_epoch_display()
        if epoch_info:
            parts.append(epoch_info)
        
        # Add time estimates if available
        time_estimate = self.get_time_estimate_display()
        if time_estimate:
            parts.append(time_estimate)
        
        time_per_epoch = self.get_time_per_epoch_display()
        if time_per_epoch:
            parts.append(time_per_epoch)
        
        return " | ".join(parts)
