import time

import torch
import typer


class Timeit:
    def __init__(self, name=None):
        """
        Initialize the context manager for time measurement.

        Args:
            name (str, optional): A name to identify this timer. If None,
                                  "Function" will be used as a default.
        """
        self.name = name if name else "Function"

    def __enter__(self):
        """
        Start the timer when entering the context.

        Returns:
            self: The context manager instance.
        """
        self.start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def now(self):
        # Handle PyTorch CUDA timing if available
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Standard Python timing
        elapsed_time = time.time() - self.start_time

        # The context manager doesn't suppress exceptions
        return elapsed_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detect_device(use_cpu: bool = False) -> str:
    # Returns the torch device string ("cpu", "cuda", or "mps")
    device = "cpu"
    if not use_cpu and torch.cuda.is_available():
        device = "cuda"
    elif not use_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device


def setup_typer(name) -> typer.Typer:
    typer_obj = typer.Typer(name=name, pretty_exceptions_show_locals=False)
    return typer_obj
