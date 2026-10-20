import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        return device

    if torch.cuda.is_available():
        print("Using CUDA device:", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")

    print("Using CPU device")
    return torch.device("cpu")