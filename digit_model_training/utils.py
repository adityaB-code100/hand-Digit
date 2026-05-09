import os
import time
import torch

def setup_directories(base_path="."):
    """Create necessary directories for saving models, logs, and reports."""
    directories = ["saved_models", "logs", "graphs", "reports"]
    for dir_name in directories:
        os.makedirs(os.path.join(base_path, dir_name), exist_ok=True)

class TimeTracker:
    """Utility to track training and inference times."""
    def __init__(self):
        self.start_time = 0
        self.end_time = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()
        return self.end_time - self.start_time

def get_model_size(model):
    """Return the model size in MB."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_bytes = os.path.getsize(f.name)
    os.remove(f.name)
    return size_bytes / (1024 * 1024)

def format_time(seconds):
    """Format time in seconds to HH:MM:SS format."""
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

def log_gpu_memory():
    """Logs GPU memory utilization."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[GPU] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
