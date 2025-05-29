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
    job_name: str = ""
    
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
    progress: float = 0.0
    error_message: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
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
        for field_name in ['start_time', 'end_time']:
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
    
    def get_progress_display(self) -> str:
        """Get progress as percentage string"""
        return f"{self.progress:.1f}%"
    
    def get_duration_display(self) -> str:
        """Get job duration if completed"""
        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        return "N/A"
