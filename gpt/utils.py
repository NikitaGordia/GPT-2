import os
import time
from typing import Callable, Optional

import torch
import typer


class Timeit:
    def __init__(self, use_cuda: bool = False) -> None:
        """
        Initialize the context manager for time measurement.

        Args:
            use_cuda (bool, optional): Whether to synchronize CUDA operations for accurate timing.
                                      Defaults to False.
        """
        self.use_cuda = use_cuda

    def __enter__(self):
        """
        Start the timer when entering the context.

        Returns:
            self: The context manager instance.
        """
        self.start_time = time.time()

        if self.use_cuda:
            torch.cuda.synchronize()

        return self

    def __exit__(self, *_) -> bool:
        return False

    def now(self) -> float:
        # Handle PyTorch CUDA timing if available
        if self.use_cuda:
            torch.cuda.synchronize()

        # Standard Python timing
        elapsed_time = time.time() - self.start_time

        # The context manager doesn't suppress exceptions
        return elapsed_time


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def detect_device(use_cpu: bool = False) -> str:
    # Returns the torch device string ("cpu", "cuda", or "mps")
    device = "cpu"
    if not use_cpu and torch.cuda.is_available():
        device = "cuda"
    elif not use_cpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device


def setup_typer(name: str) -> typer.Typer:
    typer_obj = typer.Typer(name=name, pretty_exceptions_show_locals=False)
    return typer_obj


def handle_cache_dir(cache_dir: Optional[str], sub_dir: str, log_fn: Callable[[str], None]) -> str:
    cache_dir = os.path.join(handle_env(cache_dir, "CACHE_DIR", log_fn), sub_dir)
    return cache_dir


def handle_env(value: Optional[str], key: str, log_fn: Callable[[str], None]) -> str:
    if value is None:
        value = os.environ.get(key)
        if value is None:
            raise ValueError(f"{key} not specified and environment variable not set")
        log_fn(f"Using {key} environment variable: {value}")
    return value
