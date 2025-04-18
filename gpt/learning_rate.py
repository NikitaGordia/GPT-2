import math


class CosineDecayWarmup:
    """Learning rate scheduler with linear warmup and cosine decay."""

    def __init__(self, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> None:
        """Initialize the scheduler.

        Args:
            max_lr: Maximum learning rate after warmup
            min_lr: Minimum learning rate at the end of decay
            warmup_steps: Number of warmup steps
            max_steps: Total number of steps for the schedule
        """
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

    def get_lr(self, it: int) -> float:
        """Get learning rate for the current iteration.

        Args:
            it: Current iteration number

        Returns:
            The learning rate for the current iteration
        """
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it > self.max_steps:
            return self.min_lr
        decay_ratio: float = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff: float = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
