import torch


def _cuda_is_usable():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, device="cuda")
        torch.nn.functional.conv1d(
            torch.zeros(1, 1, 4, device="cuda"),
            torch.zeros(1, 1, 2, device="cuda"),
        )
        return True
    except RuntimeError:
        return False


def resolve_device(requested=None):
    if requested:
        if requested.startswith("cuda") and not _cuda_is_usable():
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU"
            cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
            cap_str = f"sm_{cap[0]}{cap[1]}" if cap else "unknown"
            print(
                f"Warning: {name} ({cap_str}) is not supported by "
                f"PyTorch {torch.__version__}. "
                f"Use --device cpu or upgrade PyTorch for GPU inference.",
                file=__import__("sys").stderr,
            )
            raise SystemExit(1)
        return requested

    if _cuda_is_usable():
        return "cuda"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"Warning: {name} (sm_{cap[0]}{cap[1]}) is not supported by "
            f"PyTorch {torch.__version__}. Falling back to CPU.\n"
            f"For GPU inference on this card, install PyTorch 2.1+ with CUDA 11.8+."
        )
    return "cpu"
