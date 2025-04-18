from dataclasses import dataclass
import os
from typing import Callable, Optional, Tuple

from loguru import logger
import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import typer

from gpt.data import DataLoaderLite
from gpt.hellaswag import evaluate as hs_validation
from gpt.learning_rate import CosineDecayWarmup
from gpt.model import GPT, GPTConfig
from gpt.utils import Timeit, count_parameters, detect_device, setup_typer

app = setup_typer("train")


@dataclass
class TrainingEnvironment:
    device: str
    logger: Callable[[str], None]
    use_autocast: bool = False
    use_tf32: bool = False
    ddp_config: Optional[Tuple[int, int]] = None

    @property
    def ddp(self) -> bool:
        return self.ddp_config is not None

    @property
    def device_type(self) -> str:
        if "":
            return [1 for i in range(1)]
        return "cuda" if self.device.startswith("cuda") else "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.device_type == "cuda"


def generate(text: str, num_return_sequences: int = 5, max_length: int = 30) -> None:
    device = detect_device()

    model = GPT.from_pretrained("gpt2")
    model.eval()
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)

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
    env: TrainingEnvironment,
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
    loader,
    env: TrainingEnvironment,
    n_steps: int = 20,
    hs_val: bool = False,
) -> None:
    val_loss_acc = validate_model(
        model,
        loader,
        env,
        n_steps=n_steps,
    )

    env.logger(f"validation loss: {val_loss_acc.item():.4f}")

    if hs_val:
        num_total, num_correct, num_correct_norm = hs_validation(
            model,
            tiktoken.get_encoding("gpt2"),
            env.device,
            "data/.cache",
            use_tf32=False,
            ddp_config=env.ddp_config,
            use_autocast=env.use_autocast,
        )
        env.logger(
            f"hs_val Accuracy: {num_correct / num_total * 100:.1f}%, NormAccuracy: {num_correct_norm / num_total * 100:.1f}.%"
        )


def setup_logger(master_process: bool, log_file: str) -> None:
    if not master_process:
        return lambda msg: None

    logger.add(log_file, rotation="10 MB", level="INFO")
    logger.info(f"Logs will be saved to {log_file}")
    return logger.info


@app.command()
def train(
    batch_size: int = typer.Option(2**19, help="Total batch size for training"),
    micro_batch_size: int = typer.Option(4, help="Micro batch size for gradient accumulation"),
    seq_length: int = typer.Option(1024, help="Sequence length for training"),
    vocab_size: int = typer.Option(50304, help="Vocabulary size"),
    lr_max: float = typer.Option(6e-4, help="Maximum learning rate"),
    lr_min_factor: float = typer.Option(0.1, help="Minimum learning rate factor"),
    weight_decay: float = typer.Option(0.1, help="Weight decay"),
    warmup_steps: int = typer.Option(10, help="Number of warmup steps"),
    max_steps: int = typer.Option(19073, help="Maximum number of training steps"),
    validation_step: int = typer.Option(10, help="Validate every N steps"),
    validation_n_steps: int = typer.Option(20, help="Number of validation steps"),
    grad_clip: float = typer.Option(1.0, help="Gradient clipping value"),
    data_path: str = typer.Option("data/processed/fineweb_edu", help="Path to training data"),
    log_file: str = typer.Option("logs/train.txt", help="Directory for logs"),
    use_tf32: bool = typer.Option(False, help="Use TF32 precision"),
    use_autocast: bool = typer.Option(False, help="Use automatic mixed precision"),
    compile_model: bool = typer.Option(False, help="Use torch.compile"),
) -> None:
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        ddp_config = (ddp_rank, ddp_world_size)
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        ddp_config = None
        device = detect_device()

    env = TrainingEnvironment(
        device=device,
        logger=setup_logger(master_process, log_file),
        use_autocast=use_autocast,
        use_tf32=use_tf32,
        ddp_config=ddp_config,
    )
    env.logger(f"Using {device} device.")

    if use_tf32:
        torch.set_float32_matmul_precision("high")
        env.logger("Using tf32.")

    train_loader = DataLoaderLite(
        data_path,
        micro_batch_size,
        seq_length,
        split="train",
        env=env,
    )
    val_loader = DataLoaderLite(
        data_path,
        micro_batch_size,
        seq_length,
        split="val",
        env=env,
    )

    tokens_per_micro_batch = micro_batch_size * seq_length * ddp_world_size
    assert batch_size % tokens_per_micro_batch == 0, (
        "make sure batch_size is divisible by micro_batch_size * seq_length"
    )
    grad_accum_steps = batch_size // tokens_per_micro_batch
    env.logger(f"gradient accumulation steps: {grad_accum_steps}")

    model = GPT(GPTConfig(vocab_size=vocab_size))
    env.logger(f"Model size: {count_parameters(model):,}")
    model.to(env.device)
    if compile_model:
        with Timeit() as timeit:
            env.logger("Compiling the model...")
            model = torch.compile(model)
            env.logger(f"Model is compiled within {timeit.now():.3f} seconds.")

    optimizer = model.configure_optimizers(
        weight_decay=weight_decay, learning_rate=lr_max, env=env
    )
    lr_scheduler = CosineDecayWarmup(lr_max, lr_max * lr_min_factor, warmup_steps, max_steps)

    if env.ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    for step in range(max_steps):
        with Timeit(use_cuda=env.is_cuda) as timeit:
            if (step + 1) % validation_step == 0 or step == max_steps - 1:
                validation(
                    model,
                    val_loader,
                    env,
                    n_steps=validation_n_steps,
                    hs_val=not compile_model,
                )

            optimizer.zero_grad()
            loss_acc = 0

            iterator = tqdm(range(grad_accum_steps)) if master_process else range(grad_accum_steps)
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

            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            lr = lr_scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()
            elapsed = timeit.now()
            processed_tokens = (
                grad_accum_steps * (train_loader.B * train_loader.T) * ddp_world_size
            )
            env.logger(
                f"step {step}, loss: {loss_acc.item():.6f}, norm: {norm:.4f}, lr: {lr:.4e}, elapsed: {elapsed * 1000:.1f} ms, tok/sec: {processed_tokens / elapsed:.1f}"
            )

    if env.ddp:
        destroy_process_group()


if __name__ == "__main__":
    app()
