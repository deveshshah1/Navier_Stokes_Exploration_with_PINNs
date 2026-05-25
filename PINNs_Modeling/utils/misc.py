import torch

def get_device_params():
    # Choose accelerator in order: CUDA → MPS → CPU
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    if accelerator == "gpu" and devices > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    return {
        "accelerator": accelerator,
        "devices": devices,
        "strategy": strategy,
    }
