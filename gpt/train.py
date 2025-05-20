from typing import Callable

import hydra
from loguru import logger
from omegaconf import DictConfig
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from gpt.data import DataLoaderLite
from gpt.hellaswag.evaluate import evaluate as hs_validation
from gpt.learning_rate import CosineDecayWarmup
from gpt.model import GPT, GPTConfig
from gpt.utils import RuntimeEnvironment, Timeit, configure_logger, count_parameters


# @hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate(
    cfg: DictConfig,
    env: RuntimeEnvironment,
    text: str,
    num_return_sequences: int = 5,
    max_length: int = 30,
) -> None:
    model = GPT.from_pretrained("gpt2", cfg)
    model.eval()
    model.to(env.device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(env.device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)

            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)

            x = torch.cat((x, xcol), dim=1)

    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


def validate_model(
    model: nn.Module,
    loader,
    env: RuntimeEnvironment,
    n_steps: int = 20,
) -> torch.Tensor:
    model.eval()
    loader.reset()

    val_loss_acc = 0
    for _ in range(n_steps):
        X, Y = loader.next_batch()
        X, Y = X.to(env.device), Y.to(env.device)

        if env.use_autocast:
            with torch.autocast(device_type=env.device_type, dtype=torch.bfloat16):
                _, loss = model(X, Y)
        else:
            _, loss = model(X, Y)

        loss = loss / n_steps
        val_loss_acc += loss.detach()

    if env.ddp:
        dist.all_reduce(val_loss_acc, op=dist.ReduceOp.AVG)

    return val_loss_acc


@torch.no_grad()
def validation(
    model: nn.Module,
    loader: DataLoaderLite,
    env: RuntimeEnvironment,
    cfg: DictConfig,
    hs_val: bool = False,
) -> None:
    val_loss_acc = validate_model(
        model,
        loader,
        env,
        n_steps=cfg.training.validation.n_steps,
    )

    logger.info(f"validation loss: {val_loss_acc.item():.4f}")

    if hs_val:
        num_total, num_correct, num_correct_norm = hs_validation(
            cfg.data.hellaswag,
            env,
            model,
            tiktoken.get_encoding(cfg.training.validation.hellaswag.encoder),
        )
        logger.info(
            f"hs_val Accuracy: {num_correct / num_total * 100:.1f}%, NormAccuracy: {num_correct_norm / num_total * 100:.1f}.%"
        )


def setup_logger(master_process: bool, log_file: str) -> Callable[[str], None]:
    """Set up logging based on whether this is the master process.

    Args:
        master_process: Whether this is the master process
        log_file: Path to the log file

    Returns:
        A logging function that takes a string message
    """
    if not master_process:
        # Create a no-op logger for non-master processes
        return lambda _: None

    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info(f"Logs will be saved to {log_file}")
    return logger.info


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    # Setup environment
    env = RuntimeEnvironment.from_hydra_config(cfg.env)
    configure_logger(env.ddp_rank)
    logger.info(f"Using {env.device} device.")

    if env.use_tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Using tf32.")

    # Setup data loaders
    train_loader = DataLoaderLite(
        cfg.data.fineweb.path,
        cfg.training.micro_batch_size,
        cfg.training.seq_length,
        split="train",
        ddp_config=env.ddp_config,
    )
    val_loader = DataLoaderLite(
        cfg.data.fineweb.path,
        cfg.training.micro_batch_size,
        cfg.training.seq_length,
        split="val",
        ddp_config=env.ddp_config,
    )

    # Calculate gradient accumulation steps
    tokens_per_micro_batch = (
        cfg.training.micro_batch_size * cfg.training.seq_length * env.ddp_world_size
    )
    assert cfg.training.batch_size % tokens_per_micro_batch == 0, (
        "make sure batch_size is divisible by micro_batch_size * seq_length"
    )
    grad_accum_steps = cfg.training.batch_size // tokens_per_micro_batch
    logger.info(f"gradient accumulation steps: {grad_accum_steps}")

    # Setup model
    model = GPT(GPTConfig.from_hydra_config(cfg.model))
    logger.info(f"Model size: {count_parameters(model):,}")
    model.to(env.device)

    if cfg.training.compile_model:
        with Timeit() as timeit:
            logger.info("Compiling the model...")
            model = torch.compile(model)
            logger.info(f"Model is compiled within {timeit.now():.3f} seconds.")

    # Setup optimizer and learning rate scheduler
    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.optimizer.weight_decay,
        learning_rate=cfg.training.lr.max,
        device_type=env.device_type,
    )
    lr_scheduler = CosineDecayWarmup.from_hydra_config(cfg.training)

    if env.ddp:
        model = DDP(model, device_ids=[env.ddp_local_rank])

    # Training loop
    for step in range(cfg.training.max_steps):
        with Timeit(use_cuda=env.is_cuda) as timeit:
            if (
                step + 1
            ) % cfg.training.validation.step == 0 or step == cfg.training.max_steps - 1:
                validation(
                    model,
                    val_loader,
                    env,
                    cfg,
                    hs_val=cfg.training.validation.hellaswag.enabled
                    and not cfg.training.compile_model,
                )

            optimizer.zero_grad()
            loss_acc = 0

            iterator = tqdm(range(grad_accum_steps)) if env.is_master else range(grad_accum_steps)
            for it in iterator:
                X, Y = train_loader.next_batch()
                X, Y = X.to(env.device), Y.to(env.device)

                if env.ddp:
                    model.require_backward_grad_sync = it == grad_accum_steps - 1
                if env.use_autocast:
                    with torch.autocast(device_type=env.device_type, dtype=torch.bfloat16):
                        _, loss = model(X, Y)
                else:
                    _, loss = model(X, Y)
                loss = loss / grad_accum_steps
                loss_acc += loss.detach()
                loss.backward()
            if env.ddp:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.optimizer.grad_clip
            )

            lr = lr_scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            elapsed = timeit.now()
            processed_tokens = (
                grad_accum_steps * (train_loader.B * train_loader.T) * env.ddp_world_size
            )
            logger.info(
                f"step {step}, loss: {loss_acc.item():.6f}, norm: {norm:.4f}, lr: {lr:.4e}, elapsed: {elapsed * 1000:.1f} ms, tok/sec: {processed_tokens / elapsed:.1f}"
            )

    if env.ddp:
        destroy_process_group()


if __name__ == "__main__":
    train()
