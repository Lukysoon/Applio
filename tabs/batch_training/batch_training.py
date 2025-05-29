import os
import sys
import gradio as gr
from multiprocessing import cpu_count
from typing import List, Tuple, Optional

# Add current directory to path
now_dir = os.getcwd()
sys.path.append(now_dir)

from assets.i18n.i18n import I18nAuto
from .batch_manager import BatchTrainingManager
from .job_config import BatchTrainingJob

# Import utility functions from train tab
from tabs.train.train import (
    get_datasets_list,
    get_models_list,
    refresh_models_and_datasets,
    get_pretrained_list,
    refresh_custom_pretraineds,
    get_embedder_custom_list,
    refresh_custom_embedder_list,
)

from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu

i18n = I18nAuto()

# Global batch manager instance
batch_manager = BatchTrainingManager()


def create_job_from_form(*args) -> BatchTrainingJob:
    """Create a BatchTrainingJob from form inputs"""
    (
        job_name, model_name, architecture, sampling_rate, vocoder, cpu_cores, gpu,
        dataset_path, cut_preprocess, process_effects, noise_reduction, clean_strength,
        chunk_len, overlap_len, f0_method, embedder_model, embedder_model_custom,
        include_mutes, hop_length, batch_size, save_every_epoch, total_epoch,
        save_only_latest, save_every_weights, pretrained, cleanup, cache_dataset_in_gpu,
        checkpointing, custom_pretrained, g_pretrained_path, d_pretrained_path,
        overtraining_detector, overtraining_threshold, index_algorithm
    ) = args
    
    job = BatchTrainingJob(
        job_name=job_name or f"Training Job - {model_name}",
        model_name=model_name,
        architecture=architecture,
        sampling_rate=sampling_rate,
        vocoder=vocoder,
        cpu_cores=cpu_cores,
        gpu=gpu,
        dataset_path=dataset_path,
        cut_preprocess=cut_preprocess,
        process_effects=process_effects,
        noise_reduction=noise_reduction,
        clean_strength=clean_strength,
        chunk_len=chunk_len,
        overlap_len=overlap_len,
        f0_method=f0_method,
        embedder_model=embedder_model,
        embedder_model_custom=embedder_model_custom,
        include_mutes=include_mutes,
        hop_length=hop_length,
        batch_size=batch_size,
        save_every_epoch=save_every_epoch,
        total_epoch=total_epoch,
        save_only_latest=save_only_latest,
        save_every_weights=save_every_weights,
        pretrained=pretrained,
        cleanup=cleanup,
        cache_dataset_in_gpu=cache_dataset_in_gpu,
        checkpointing=checkpointing,
        custom_pretrained=custom_pretrained,
        g_pretrained_path=g_pretrained_path,
        d_pretrained_path=d_pretrained_path,
        overtraining_detector=overtraining_detector,
        overtraining_threshold=overtraining_threshold,
        index_algorithm=index_algorithm,
    )
    
    return job


def add_job_to_queue(*args) -> Tuple[str, str]:
    """Add a new job to the batch queue"""
    try:
        job = create_job_from_form(*args)
        
        # Validate required fields
        if not job.dataset_name.strip():
            return "❌ Error: Model name is required", get_job_queue_display()
        
        if not job.dataset_path.strip():
            return "❌ Error: Dataset path is required", get_job_queue_display()
        
        # Check if model name already exists in queue
        for existing_job in batch_manager.jobs:
            if existing_job.model_name == job.model_name and existing_job.status == "pending":
                return f"❌ Error: Job with model name '{job.model_name}' already exists in queue", get_job_queue_display()
        
        batch_manager.add_job(job)
        return f"✅ Added job: {job.job_name}", get_job_queue_display()
        
    except Exception as e:
        return f"❌ Error adding job: {str(e)}", get_job_queue_display()


def remove_job_from_queue(job_index: int) -> Tuple[str, str]:
    """Remove a job from the queue"""
    try:
        if 0 <= job_index < len(batch_manager.jobs):
            job = batch_manager.jobs[job_index]
            if batch_manager.remove_job(job.job_id):
                return f"✅ Removed job: {job.job_name}", get_job_queue_display()
            else:
                return f"❌ Cannot remove job: {job.job_name} (not in pending status)", get_job_queue_display()
        else:
            return "❌ Invalid job index", get_job_queue_display()
    except Exception as e:
        return f"❌ Error removing job: {str(e)}", get_job_queue_display()


def move_job_in_queue(job_index: int, direction: str) -> Tuple[str, str]:
    """Move a job up or down in the queue"""
    try:
        if 0 <= job_index < len(batch_manager.jobs):
            job = batch_manager.jobs[job_index]
            if batch_manager.move_job(job.job_id, direction):
                return f"✅ Moved job: {job.job_name} {direction}", get_job_queue_display()
            else:
                return f"❌ Cannot move job: {job.job_name}", get_job_queue_display()
        else:
            return "❌ Invalid job index", get_job_queue_display()
    except Exception as e:
        return f"❌ Error moving job: {str(e)}", get_job_queue_display()


def clear_job_queue() -> Tuple[str, str]:
    """Clear all pending jobs from the queue"""
    try:
        batch_manager.clear_queue()
        return "✅ Cleared all pending jobs from queue", get_job_queue_display()
    except Exception as e:
        return f"❌ Error clearing queue: {str(e)}", get_job_queue_display()


def start_batch_execution() -> str:
    """Start batch execution"""
    try:
        batch_manager.start_batch()
        return "✅ Batch execution started"
    except Exception as e:
        return f"❌ Error starting batch: {str(e)}"


def pause_batch_execution() -> str:
    """Pause batch execution"""
    try:
        if batch_manager.is_paused:
            batch_manager.resume_batch()
            return "✅ Batch execution resumed"
        else:
            batch_manager.pause_batch()
            return "✅ Batch execution paused"
    except Exception as e:
        return f"❌ Error pausing/resuming batch: {str(e)}"


def stop_batch_execution() -> str:
    """Stop batch execution"""
    try:
        batch_manager.stop_batch()
        return "✅ Batch execution stopped"
    except Exception as e:
        return f"❌ Error stopping batch: {str(e)}"


def get_job_queue_display() -> str:
    """Get formatted display of job queue"""
    if not batch_manager.jobs:
        return "No jobs in queue"
    
    lines = []
    for i, job in enumerate(batch_manager.jobs):
        status_icon = {
            "pending": "⏳",
            "preprocessing": "🔄",
            "extracting": "🔍",
            "training": "🎯",
            "indexing": "📊",
            "completed": "✅",
            "failed": "❌"
        }.get(job.status, "❓")
        
        line = f"{i+1}. {status_icon} {job.job_name}"
        line += f" | Model: {job.model_name}"
        line += f" | Dataset: {os.path.basename(job.dataset_path) if job.dataset_path else 'N/A'}"
        line += f" | SR: {job.sampling_rate}Hz"
        line += f" | Epochs: {job.total_epoch}"
        
        if job.status not in ["pending", "completed", "failed"]:
            line += f" | Progress: {job.get_progress_display()}"
        
        if job.status == "failed" and job.error_message:
            line += f" | Error: {job.error_message[:50]}..."
        
        lines.append(line)
    
    return "\n".join(lines)


def get_batch_status_display() -> Tuple[str, str, str]:
    """Get batch execution status display"""
    if batch_manager.is_running:
        if batch_manager.is_paused:
            status = "⏸️ Paused"
        else:
            status = "🔄 Running"
    else:
        status = "⏹️ Stopped"
    
    completed, total, percentage = batch_manager.get_overall_progress()
    progress = f"Progress: {completed}/{total} jobs ({percentage:.1f}%)"
    
    current_job = batch_manager.get_current_job_info()
    current = f"Current: {current_job}" if current_job else "Current: None"
    
    return status, progress, current


def get_execution_log_display() -> str:
    """Get execution log display"""
    log_entries = batch_manager.get_execution_log()
    if not log_entries:
        return "No log entries"
    
    # Return last 20 entries
    return "\n".join(log_entries[-20:])


def save_batch_configuration(filepath: str) -> str:
    """Save batch configuration to file"""
    try:
        if not filepath.endswith('.json'):
            filepath += '.json'
        
        batch_manager.save_batch_config(filepath)
        return f"✅ Configuration saved to {filepath}"
    except Exception as e:
        return f"❌ Error saving configuration: {str(e)}"


def load_batch_configuration(filepath: str) -> Tuple[str, str]:
    """Load batch configuration from file"""
    try:
        if batch_manager.load_batch_config(filepath):
            return f"✅ Configuration loaded from {filepath}", get_job_queue_display()
        else:
            return f"❌ Failed to load configuration from {filepath}", get_job_queue_display()
    except Exception as e:
        return f"❌ Error loading configuration: {str(e)}", get_job_queue_display()


def batch_training_tab():
    """Create the batch training tab interface"""
    
    # Set up progress callback
    def update_displays():
        return (
            get_job_queue_display(),
            *get_batch_status_display(),
            get_execution_log_display()
        )
    
    batch_manager.set_progress_callback(lambda: None)  # Will be updated with actual components
    
    with gr.Column():
        gr.Markdown("# Batch Training")
        gr.Markdown("Configure and run multiple training jobs in sequence. Each job will run the complete training pipeline: preprocess → extract → train → index.")
        
        # Job Configuration Section
        with gr.Accordion("Add New Job", open=True):
            with gr.Row():
                with gr.Column():

                    model_name = gr.Dropdown(
                        label="Experiment Name",
                        choices=get_datasets_list(),
                        value=None,
                        interactive=True,
                        allow_custom_value=False,
                        visible=False
                    )

                    architecture = gr.Radio(
                        label="Architecture",
                        choices=["RVC", "Applio"],
                        value="RVC",
                        interactive=True,
                        visible=False,
                    )
                
                with gr.Column():
                    sampling_rate = gr.Radio(
                        label="Sampling Rate",
                        choices=["32000", "40000", "48000"],
                        value="48000",
                        interactive=True,
                        visible=False
                    )
                    vocoder = gr.Radio(
                        label="Vocoder",
                        choices=["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"],
                        value="HiFi-GAN",
                        interactive=False,
                        visible=False,
                    )
            
            with gr.Row():
                with gr.Column():
                    cpu_cores = gr.Slider(
                        1,
                        min(cpu_count(), 32),
                        min(cpu_count(), 32),
                        step=1,
                        label="CPU Cores",
                        interactive=False,
                    )
                
                with gr.Column():
                    gpu = gr.Textbox(
                        label="GPU Number",
                        placeholder="0 to ∞ separated by -",
                        value=str(get_number_of_gpus()),
                        interactive=False,
                    )
            
            # Dataset and Preprocessing
            with gr.Accordion("Dataset & Preprocessing", open=False):
                dataset_path = gr.Dropdown(
                    label="Dataset Path",
                    choices=get_datasets_list(),
                    allow_custom_value=False,
                    interactive=True,
                )
                
                with gr.Row():
                    cut_preprocess = gr.Radio(
                        label="Audio cutting",
                        choices=["Skip", "Simple", "Automatic"],
                        value="Automatic",
                        interactive=False,
                        visible=False
                        
                    )
                    process_effects = gr.Checkbox(
                        label="Process effects",
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    noise_reduction = gr.Checkbox(
                        label="Noise Reduction",
                        value=False,
                        interactive=True,
                        visible=False
                    )
                
                with gr.Row():
                    clean_strength = gr.Slider(
                        0, 1, 0.5, step=0.1,
                        label="Noise Reduction Strength",
                        interactive=True,
                        visible=False
                    )
                    chunk_len = gr.Slider(
                        0.5, 5.0, 3.0, step=0.1,
                        label="Chunk length (sec)",
                        interactive=False,
                    )
                    overlap_len = gr.Slider(
                        0.0, 0.4, 0.3, step=0.1,
                        label="Overlap length (sec)",
                        interactive=False,
                    )
            
            # Extract Settings
            with gr.Accordion("Extract Settings", open=False):
                with gr.Row():
                    f0_method = gr.Radio(
                        label="Pitch extraction algorithm",
                        choices=["crepe", "crepe-tiny", "rmvpe"],
                        value="rmvpe",
                        interactive=False,
                    )
                    embedder_model = gr.Radio(
                        label="Embedder Model",
                        choices=["contentvec", "custom"],
                        value="contentvec",
                        interactive=True,
                    )
                
                with gr.Row():
                    include_mutes = gr.Slider(
                        0, 10, 2, step=1,
                        label="Silent training files",
                        interactive=True,
                    )
                    hop_length = gr.Slider(
                        1, 512, 128, step=1,
                        label="Hop Length",
                        visible=False,
                        interactive=True,
                    )
                
                embedder_model_custom = gr.Dropdown(
                    label="Custom Embedder",
                    choices=get_embedder_custom_list(),
                    interactive=True,
                    allow_custom_value=True,
                    visible=False,
                )
            
            # Training Settings
            with gr.Accordion("Training Settings", open=False):
                with gr.Row():
                    batch_size = gr.Slider(
                        1, 50, max_vram_gpu(0), step=1,
                        label="Batch Size",
                        interactive=True,
                    )
                    save_every_epoch = gr.Slider(
                        1, 100, 10, step=1,
                        label="Save Every Epoch",
                        interactive=True,
                    )
                    total_epoch = gr.Slider(
                        1, 10000, 500, step=1,
                        label="Total Epoch",
                        interactive=True,
                    )
                
                with gr.Row():
                    save_only_latest = gr.Checkbox(
                        label="Save Only Latest",
                        value=True,
                        interactive=True,
                    )
                    save_every_weights = gr.Checkbox(
                        label="Save Every Weights",
                        value=True,
                        interactive=True,
                    )
                    pretrained = gr.Checkbox(
                        label="Pretrained",
                        value=True,
                        interactive=True,
                    )
                
                with gr.Row():
                    cleanup = gr.Checkbox(
                        label="Fresh Training",
                        value=False,
                        interactive=True,
                    )
                    cache_dataset_in_gpu = gr.Checkbox(
                        label="Cache Dataset in GPU",
                        value=False,
                        interactive=True,
                    )
                    checkpointing = gr.Checkbox(
                        label="Checkpointing",
                        value=False,
                        interactive=True,
                    )
                
                # Advanced Training Settings
                with gr.Row():
                    custom_pretrained = gr.Checkbox(
                        label="Custom Pretrained",
                        value=False,
                        interactive=True,
                    )
                    overtraining_detector = gr.Checkbox(
                        label="Overtraining Detector",
                        value=False,
                        interactive=True,
                    )
                
                with gr.Row(visible=False) as pretrained_custom_settings:
                    g_pretrained_path = gr.Dropdown(
                        label="Custom Pretrained G",
                        choices=get_pretrained_list("G"),
                        interactive=True,
                        allow_custom_value=True,
                    )
                    d_pretrained_path = gr.Dropdown(
                        label="Custom Pretrained D",
                        choices=get_pretrained_list("D"),
                        interactive=True,
                        allow_custom_value=True,
                    )
                
                with gr.Row(visible=False) as overtraining_settings:
                    overtraining_threshold = gr.Slider(
                        1, 100, 50, step=1,
                        label="Overtraining Threshold",
                        interactive=True,
                    )
                
                index_algorithm = gr.Radio(
                    label="Index Algorithm",
                    choices=["Auto", "Faiss", "KMeans"],
                    value="Auto",
                    interactive=True,
                )
            
            # Add Job Button
            with gr.Row():
                add_job_btn = gr.Button("Add Job to Queue", variant="primary")
                refresh_btn = gr.Button("Refresh Lists")
            
            add_job_status = gr.Textbox(
                label="Status",
                interactive=False,
                max_lines=2,
            )
        
        # Job Queue Management
        with gr.Accordion("Job Queue", open=True):
            job_queue_display = gr.Textbox(
                label="Queued Jobs",
                value=get_job_queue_display(),
                interactive=False,
                max_lines=10,
            )
            
            with gr.Row():
                job_index_input = gr.Number(
                    label="Job Index (1-based)",
                    value=1,
                    minimum=1,
                    interactive=True,
                )
                remove_job_btn = gr.Button("Remove Job")
                move_up_btn = gr.Button("Move Up")
                move_down_btn = gr.Button("Move Down")
                clear_queue_btn = gr.Button("Clear Queue", variant="stop")
            
            queue_status = gr.Textbox(
                label="Queue Status",
                interactive=False,
                max_lines=2,
            )
        
        # Batch Configuration
        with gr.Accordion("Batch Configuration", open=False):
            with gr.Row():
                config_filepath = gr.Textbox(
                    label="Configuration File Path",
                    placeholder="batch_config.json",
                    interactive=True,
                )
                save_config_btn = gr.Button("Save Config")
                load_config_btn = gr.Button("Load Config")
            
            config_status = gr.Textbox(
                label="Configuration Status",
                interactive=False,
                max_lines=2,
            )
        
        # Batch Execution
        with gr.Accordion("Batch Execution", open=True):
            with gr.Row():
                batch_status_display = gr.Textbox(
                    label="Status",
                    value=get_batch_status_display()[0],
                    interactive=False,
                )
                batch_progress_display = gr.Textbox(
                    label="Progress",
                    value=get_batch_status_display()[1],
                    interactive=False,
                )
                current_job_display = gr.Textbox(
                    label="Current Job",
                    value=get_batch_status_display()[2],
                    interactive=False,
                )
            
            with gr.Row():
                start_batch_btn = gr.Button("Start Batch", variant="primary")
                pause_resume_btn = gr.Button("Pause/Resume")
                stop_batch_btn = gr.Button("Stop Batch", variant="stop")
            
            execution_status = gr.Textbox(
                label="Execution Status",
                interactive=False,
                max_lines=2,
            )
            
            execution_log = gr.Textbox(
                label="Execution Log",
                value=get_execution_log_display(),
                interactive=False,
                max_lines=15,
            )
        
        # Event handlers
        def toggle_custom_pretrained(custom_pretrained_enabled):
            return {"visible": custom_pretrained_enabled, "__type__": "update"}
        
        def toggle_overtraining_settings(overtraining_enabled):
            return {"visible": overtraining_enabled, "__type__": "update"}
        
        def toggle_embedder_custom(embedder_model_selected):
            return {"visible": embedder_model_selected == "custom", "__type__": "update"}
        
        def toggle_hop_length(f0_method_selected):
            return {"visible": f0_method_selected in ["crepe", "crepe-tiny"], "__type__": "update"}
        
        # Wire up event handlers
        custom_pretrained.change(
            fn=toggle_custom_pretrained,
            inputs=[custom_pretrained],
            outputs=[pretrained_custom_settings],
        )
        
        overtraining_detector.change(
            fn=toggle_overtraining_settings,
            inputs=[overtraining_detector],
            outputs=[overtraining_settings],
        )
        
        embedder_model.change(
            fn=toggle_embedder_custom,
            inputs=[embedder_model],
            outputs=[embedder_model_custom],
        )
        
        f0_method.change(
            fn=toggle_hop_length,
            inputs=[f0_method],
            outputs=[hop_length],
        )

        job_name = model_name
        
        # Job management event handlers
        add_job_btn.click(
            fn=add_job_to_queue,
            inputs=[
                job_name, model_name, architecture, sampling_rate, vocoder, cpu_cores, gpu,
                dataset_path, cut_preprocess, process_effects, noise_reduction, clean_strength,
                chunk_len, overlap_len, f0_method, embedder_model, embedder_model_custom,
                include_mutes, hop_length, batch_size, save_every_epoch, total_epoch,
                save_only_latest, save_every_weights, pretrained, cleanup, cache_dataset_in_gpu,
                checkpointing, custom_pretrained, g_pretrained_path, d_pretrained_path,
                overtraining_detector, overtraining_threshold, index_algorithm
            ],
            outputs=[add_job_status, job_queue_display],
        )
        
        refresh_btn.click(
            fn=refresh_models_and_datasets,
            inputs=[],
            outputs=[model_name, dataset_path],
        )
        
        remove_job_btn.click(
            fn=lambda idx: remove_job_from_queue(int(idx) - 1),
            inputs=[job_index_input],
            outputs=[queue_status, job_queue_display],
        )
        
        move_up_btn.click(
            fn=lambda idx: move_job_in_queue(int(idx) - 1, "up"),
            inputs=[job_index_input],
            outputs=[queue_status, job_queue_display],
        )
        
        move_down_btn.click(
            fn=lambda idx: move_job_in_queue(int(idx) - 1, "down"),
            inputs=[job_index_input],
            outputs=[queue_status, job_queue_display],
        )
        
        clear_queue_btn.click(
            fn=clear_job_queue,
            inputs=[],
            outputs=[queue_status, job_queue_display],
        )
        
        # Configuration event handlers
        save_config_btn.click(
            fn=save_batch_configuration,
            inputs=[config_filepath],
            outputs=[config_status],
        )
        
        load_config_btn.click(
            fn=load_batch_configuration,
            inputs=[config_filepath],
            outputs=[config_status, job_queue_display],
        )
        
        # Batch execution event handlers
        start_batch_btn.click(
            fn=start_batch_execution,
            inputs=[],
            outputs=[execution_status],
        )
        
        pause_resume_btn.click(
            fn=pause_batch_execution,
            inputs=[],
            outputs=[execution_status],
        )
        
        stop_batch_btn.click(
            fn=stop_batch_execution,
            inputs=[],
            outputs=[execution_status],
        )
        
        # Auto-refresh displays every 2 seconds when batch is running
        def auto_refresh():
            return update_displays()
        
        # Set up auto-refresh timer
        refresh_timer = gr.Timer(value=2.0)
        refresh_timer.tick(
            fn=auto_refresh,
            outputs=[job_queue_display, batch_status_display, batch_progress_display, current_job_display, execution_log],
        )
