import json
import os
import threading
import time
from datetime import datetime
from typing import List, Optional, Callable, Tuple
from .job_config import BatchTrainingJob

# Import core training functions
from core import (
    run_preprocess_script,
    run_extract_script,
    run_train_script,
    run_index_script,
)


class BatchTrainingManager:
    """Manages batch training job execution"""
    
    def __init__(self):
        self.jobs: List[BatchTrainingJob] = []
        self.current_job_index: int = 0
        self.is_running: bool = False
        self.is_paused: bool = False
        self.execution_log: List[str] = []
        self.progress_callback: Optional[Callable] = None
        self.log_callback: Optional[Callable] = None
        self._execution_thread: Optional[threading.Thread] = None
        
    def add_job(self, job: BatchTrainingJob) -> None:
        """Add a job to the batch queue"""
        self.jobs.append(job)
        self._log(f"Added job: {job.job_name}")
    
    def remove_job(self, job_id: str) -> bool:
        """Remove a job from the queue"""
        for i, job in enumerate(self.jobs):
            if job.job_id == job_id:
                if job.status == "pending":
                    removed_job = self.jobs.pop(i)
                    self._log(f"Removed job: {removed_job.job_name}")
                    return True
                else:
                    self._log(f"Cannot remove job {job.job_name} - not in pending status")
                    return False
        return False
    
    def get_job_by_id(self, job_id: str) -> Optional[BatchTrainingJob]:
        """Get a job by its ID"""
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def clear_queue(self) -> None:
        """Clear all pending jobs from the queue"""
        pending_jobs = [job for job in self.jobs if job.status == "pending"]
        self.jobs = [job for job in self.jobs if job.status != "pending"]
        self._log(f"Cleared {len(pending_jobs)} pending jobs from queue")
    
    def start_batch(self) -> None:
        """Start batch execution in a separate thread"""
        if self.is_running:
            self._log("Batch is already running")
            return
        
        if not self.jobs:
            self._log("No jobs in queue to execute")
            return
        
        pending_jobs = [job for job in self.jobs if job.status == "pending"]
        if not pending_jobs:
            self._log("No pending jobs to execute")
            return
        
        self.is_running = True
        self.is_paused = False
        self.current_job_index = 0
        
        # Find first pending job
        for i, job in enumerate(self.jobs):
            if job.status == "pending":
                self.current_job_index = i
                break
        
        self._log(f"Starting batch execution with {len(pending_jobs)} jobs")
        
        # Start execution in separate thread
        self._execution_thread = threading.Thread(target=self._execute_batch_thread)
        self._execution_thread.daemon = True
        self._execution_thread.start()
    
    def pause_batch(self) -> None:
        """Pause batch execution"""
        if self.is_running:
            self.is_paused = True
            self._log("Batch execution paused")
    
    def resume_batch(self) -> None:
        """Resume batch execution"""
        if self.is_running and self.is_paused:
            self.is_paused = False
            self._log("Batch execution resumed")
    
    def stop_batch(self) -> None:
        """Stop batch execution"""
        if self.is_running:
            self.is_running = False
            self.is_paused = False
            self._log("Batch execution stopped")
    
    def _execute_batch_thread(self) -> None:
        """Execute batch in separate thread"""
        try:
            for i in range(self.current_job_index, len(self.jobs)):
                if not self.is_running:
                    break
                
                job = self.jobs[i]
                if job.status != "pending":
                    continue
                
                # Wait if paused
                while self.is_paused and self.is_running:
                    time.sleep(0.5)
                
                if not self.is_running:
                    break
                
                self.current_job_index = i
                success = self._execute_single_job(job)
                
                if not success:
                    self._log(f"Job {job.job_name} failed. Stopping batch execution.")
                    break
                
                self._update_progress()
            
            self.is_running = False
            self.is_paused = False
            self._log("Batch execution completed")
            
        except Exception as e:
            self.is_running = False
            self.is_paused = False
            self._log(f"Batch execution failed with error: {str(e)}")
    
    def _execute_single_job(self, job: BatchTrainingJob) -> bool:
        """Execute a single training job through all pipeline steps"""
        try:
            job.start_time = datetime.now()
            job.status = "preprocessing"
            job.current_epoch = 0
            job.first_epoch_start_time = None
            job.first_epoch_end_time = None
            job.estimated_total_time = None
            job.time_per_epoch = None
            self._log(f"Starting job: {job.job_name}")
            self._update_progress()
            
            # Step 1: Preprocessing
            job.current_step = "Preprocessing dataset"
            self._update_progress()
            
            preprocess_result = run_preprocess_script(
                model_name=job.model_name,
                dataset_path=job.dataset_path,
                sample_rate=int(job.sampling_rate),
                cpu_cores=job.cpu_cores,
                cut_preprocess=job.cut_preprocess,
                process_effects=job.process_effects,
                noise_reduction=job.noise_reduction,
                clean_strength=job.clean_strength,
                chunk_len=job.chunk_len,
                overlap_len=job.overlap_len,
            )
            
            if not self.is_running:
                return False
            
            # Step 2: Feature Extraction
            job.status = "extracting"
            job.current_step = "Extracting features"
            self._update_progress()
            
            print("MODEL:")
            print(job.embedder_model)
            print(job.embedder_model_custom)

            extract_result = run_extract_script(
                model_name=job.model_name,
                f0_method=job.f0_method,
                hop_length=job.hop_length,
                cpu_cores=job.cpu_cores,
                gpu=job.gpu,
                sample_rate=int(job.sampling_rate),
                embedder_model=job.embedder_model,
                embedder_model_custom=job.embedder_model_custom,
                include_mutes=job.include_mutes,
            )
            
            if not self.is_running:
                return False
            
            # Step 3: Training
            job.status = "training"
            job.current_step = "Training model"
            job.current_epoch = 1
            job.first_epoch_start_time = datetime.now()
            self._update_progress()
            
            # Start training with progress monitoring
            train_result = self._run_train_with_monitoring(job)
            
            if not self.is_running:
                return False
            
            # Step 4: Index Generation (already included in train_script, but we'll show progress)
            job.status = "indexing"
            job.current_step = "Generating index"
            self._update_progress()
            
            # Small delay to show indexing step
            time.sleep(1)
            
            # Job completed successfully
            job.status = "completed"
            job.current_step = "Completed"
            job.end_time = datetime.now()
            self._log(f"Job {job.job_name} completed successfully")
            self._update_progress()
            
            return True
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.end_time = datetime.now()
            self._log(f"Job {job.job_name} failed: {str(e)}")
            self._update_progress()
            return False
    
    def get_overall_progress(self) -> Tuple[int, int, float]:
        """Get overall batch progress"""
        completed_jobs = len([job for job in self.jobs if job.status == "completed"])
        total_jobs = len(self.jobs)
        
        if total_jobs == 0:
            return 0, 0
        
        # Add current job progress if running
        if self.is_running and self.current_job_index < len(self.jobs):
            current_job = self.jobs[self.current_job_index]
        
        return completed_jobs, total_jobs
    
    def get_remainng_time(self) -> Optional[str]:
        """Get information about the currently running job"""
        if not self.is_running or self.current_job_index >= len(self.jobs):
            return None
        
        current_job = self.jobs[self.current_job_index]
        if current_job.status in ["completed", "failed"]:
            return None
        
        return f"{current_job.get_duration_display()}"
    
    def get_current_job_info(self) -> Optional[str]:
        """Get information about the currently running job"""
        if not self.is_running or self.current_job_index >= len(self.jobs):
            return None
        
        current_job = self.jobs[self.current_job_index]
        if current_job.status in ["completed", "failed"]:
            return None
        
        return f"{current_job.job_name}: {current_job.current_step}"
    
    def save_batch_config(self, filepath: str) -> None:
        """Save batch configuration to file"""
        try:
            config = {
                "jobs": [job.to_dict() for job in self.jobs],
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self._log(f"Batch configuration saved to {filepath}")
            
        except Exception as e:
            self._log(f"Failed to save batch configuration: {str(e)}")
    
    def load_batch_config(self, filepath: str) -> bool:
        """Load batch configuration from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Clear current jobs and load from config
            self.jobs.clear()
            for job_data in config.get("jobs", []):
                job = BatchTrainingJob.from_dict(job_data)
                # Reset status to pending for reloaded jobs
                if job.status not in ["completed"]:
                    job.status = "pending"
                    job.error_message = ""
                    job.start_time = None
                    job.end_time = None
                self.jobs.append(job)
            
            self._log(f"Loaded {len(self.jobs)} jobs from {filepath}")
            return True
            
        except Exception as e:
            self._log(f"Failed to load batch configuration: {str(e)}")
            return False
    
    def set_progress_callback(self, callback: Callable) -> None:
        """Set callback for progress updates"""
        self.progress_callback = callback
    
    def set_log_callback(self, callback: Callable) -> None:
        """Set callback for log updates"""
        self.log_callback = callback
    
    def _update_progress(self) -> None:
        """Trigger progress update callback"""
        if self.progress_callback:
            self.progress_callback()
    
    def _log(self, message: str) -> None:
        """Add message to execution log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.execution_log.append(log_entry)
        
        # Keep only last 100 log entries
        if len(self.execution_log) > 100:
            self.execution_log = self.execution_log[-100:]
        
        if self.log_callback:
            self.log_callback(log_entry)
    
    def get_execution_log(self) -> List[str]:
        """Get the execution log"""
        return self.execution_log.copy()
    
    def _run_train_with_monitoring(self, job: BatchTrainingJob) -> str:
        """Run training with real epoch progress monitoring"""
        import subprocess
        import threading
        
        # Start training in a separate thread so we can monitor progress
        training_thread = threading.Thread(
            target=self._run_training_subprocess,
            args=(job,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Monitor progress while training runs
        self._monitor_training_progress(job)
        
        # Wait for training to complete
        training_thread.join()
        
        return f"Model {job.model_name} trained successfully."
    
    def _run_training_subprocess(self, job: BatchTrainingJob) -> None:
        """Run the actual training subprocess"""
        try:
            train_result = run_train_script(
                model_name=job.model_name,
                save_every_epoch=job.save_every_epoch,
                save_only_latest=job.save_only_latest,
                save_every_weights=job.save_every_weights,
                total_epoch=job.total_epoch,
                sample_rate=int(job.sampling_rate),
                batch_size=job.batch_size,
                gpu=job.gpu,
                overtraining_detector=job.overtraining_detector,
                overtraining_threshold=job.overtraining_threshold,
                pretrained=job.pretrained,
                cleanup=job.cleanup,
                index_algorithm=job.index_algorithm,
                cache_data_in_gpu=job.cache_dataset_in_gpu,
                custom_pretrained=job.custom_pretrained,
                g_pretrained_path=job.g_pretrained_path,
                d_pretrained_path=job.d_pretrained_path,
                vocoder=job.vocoder,
                checkpointing=job.checkpointing,
            )
        except Exception as e:
            self._log(f"Training subprocess failed: {str(e)}")
    
    def _monitor_training_progress(self, job: BatchTrainingJob) -> None:
        """Monitor training progress by reading the progress file"""
        from rvc.train.progress_monitor import TrainingProgressMonitor
        
        progress_file = os.path.join("logs", job.model_name, "training_progress.json")
        last_epoch = 0
        
        # Wait for training to start and create progress file
        start_time = time.time()
        while not os.path.exists(progress_file) and self.is_running:
            time.sleep(1)
            # Timeout after 30 seconds if progress file doesn't appear
            if time.time() - start_time > 30:
                self._log(f"Warning: Progress file not found for {job.job_name}, using fallback monitoring")
                self._fallback_progress_monitoring(job)
                return
        
        # Monitor progress while training runs
        while self.is_running:
            try:
                progress_data = TrainingProgressMonitor.read_progress(job.model_name)
                
                if progress_data:
                    current_epoch = progress_data.get("current_epoch", 0)
                    status = progress_data.get("status", "training")
                    progress_percentage = progress_data.get("progress_percentage", 0)
                    current_step = progress_data.get("current_step", "Training")
                    time_per_epoch = progress_data.get("time_per_epoch")
                    estimated_remaining = progress_data.get("estimated_time_remaining")

                    # Update job with real progress data
                    job.current_epoch = current_epoch
                    job.current_step = current_step
                    
                    # Update timing information
                    if time_per_epoch and not job.time_per_epoch:
                        job.time_per_epoch = time_per_epoch
                        if not job.first_epoch_end_time and current_epoch >= 1:
                            job.first_epoch_end_time = datetime.now()
                            job.calculate_time_estimates()
                    
                    if estimated_remaining:
                        job.estimated_total_time = estimated_remaining
                    
                    # Log epoch changes
                    if current_epoch > last_epoch:
                        last_epoch = current_epoch
                        time_display = job.get_time_estimate_display() if job.time_per_epoch else "Calculating..."
                        self._log(f"Job {job.job_name}: Epoch {current_epoch}/{job.total_epoch} - {time_display}")
                    
                    # Check if training is completed
                    if status == "completed":
                        self._log(f"Training completed for {job.job_name}")
                        break
                    
                    # Check for errors
                    if status == "error":
                        error_msg = progress_data.get("error_message", "Unknown error")
                        self._log(f"Training error for {job.job_name}: {error_msg}")
                        break
                    
                    self._update_progress()
                
                time.sleep(2)  # Check progress every 2 seconds
                
            except Exception as e:
                self._log(f"Error reading progress for {job.job_name}: {str(e)}")
                time.sleep(5)  # Wait longer on error
        
        # Clean up progress file after training
        try:
            TrainingProgressMonitor.cleanup_progress_file(job.model_name)
        except Exception:
            pass  # Ignore cleanup errors
    
    def _fallback_progress_monitoring(self, job: BatchTrainingJob) -> None:
        """Fallback progress monitoring when progress file is not available"""
        self._log(f"Using fallback progress monitoring for {job.job_name}")
        
        # Simple time-based estimation
        estimated_epoch_duration = 60  # Assume 60 seconds per epoch as fallback
        
        for epoch in range(1, job.total_epoch + 1):
            if not self.is_running:
                break
            
            job.current_epoch = epoch
            job.current_step = f"Training model - Epoch {epoch}/{job.total_epoch}"
            
            # Update timing estimates
            if epoch == 1:
                job.first_epoch_start_time = datetime.now()
            elif epoch == 2:
                job.first_epoch_end_time = datetime.now()
                job.calculate_time_estimates()
            
            if job.time_per_epoch:
                remaining_epochs = job.total_epoch - epoch
                job.estimated_total_time = remaining_epochs * job.time_per_epoch
            
            self._update_progress()
            self._log(f"Job {job.job_name}: Epoch {epoch}/{job.total_epoch} (fallback monitoring)")
            
            # Wait for estimated epoch duration
            time.sleep(min(estimated_epoch_duration, 10))  # Cap at 10 seconds for demo
    
    def _simulate_epoch_progress(self, job: BatchTrainingJob) -> None:
        """Simulate epoch progression for demonstration purposes"""
        # This is a simplified simulation - in reality, you'd monitor actual training progress
        epoch_duration = 30  # Simulate 30 seconds per epoch for demo
        
        for epoch in range(1, min(6, job.total_epoch + 1)):  # Simulate first 5 epochs
            if not self.is_running:
                break
                
            job.current_epoch = epoch
            
            # Record first epoch timing
            if epoch == 1:
                job.first_epoch_start_time = datetime.now()
            elif epoch == 2 and job.first_epoch_start_time:
                job.first_epoch_end_time = datetime.now()
                job.calculate_time_estimates()
                self._log(f"First epoch completed for {job.job_name}. Estimated total time: {job.get_time_estimate_display()}")
            
            # Update progress based on epoch
            job.current_step = f"Training model - Epoch {epoch}/{job.total_epoch}"
            
            # Update time estimates for subsequent epochs
            if job.time_per_epoch and epoch > 1:
                remaining_epochs = job.total_epoch - epoch
                job.estimated_total_time = remaining_epochs * job.time_per_epoch
            
            self._update_progress()
            self._log(f"Job {job.job_name}: {job.get_epoch_display()} - {job.get_time_estimate_display()}")
            
            # Simulate epoch duration
            time.sleep(2)  # Shortened for demo - real epochs take much longer
        
        # Complete the training simulation
        job.current_epoch = job.total_epoch
        job.current_step = f"Training completed - {job.total_epoch} epochs"
        self._update_progress()
