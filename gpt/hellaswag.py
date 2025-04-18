import json
import os
from typing import Any, Callable, Dict, Generator, List, Tuple

from loguru import logger
import requests
import tiktoken
from tiktoken import Encoding
import torch
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel
import typer

from gpt.utils import setup_typer

# URLs for the HellaSwag dataset files
HELLASWAGS: Dict[str, str] = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

app = setup_typer("hellaswag")


def download_file(url: str, fname: str, chunk_size: int = 1024) -> None:
    """Helper function to download a file from a given url.

    Args:
        url: URL to download from
        fname: Local filename to save to
        chunk_size: Size of chunks to download at a time
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with (
        open(fname, "wb") as file,
        tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split: str, cache_dir: str, ddp_rank: int = 0) -> str:
    """Download a specific split of the HellaSwag dataset if not already cached.

    Args:
        split: Dataset split to download ('train', 'val', or 'test')
        cache_dir: Directory to cache the downloaded files
        ddp_rank: Current process rank in DDP setup

    Returns:
        Path to the downloaded/cached file
    """
    os.makedirs(cache_dir, exist_ok=True)
    data_url = HELLASWAGS[split]

    data_filename = os.path.join(cache_dir, f"hellaswag_{split}.json")
    if not os.path.exists(data_filename) and ddp_rank == 0:  # Only download on rank 0
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

    if dist.is_initialized():
        dist.barrier()
    return data_filename


def iterate_examples(split: str, cache_dir: str) -> Generator[Dict[str, Any], None, None]:
    """Iterate through examples in a specific split of the HellaSwag dataset.

    Args:
        split: Dataset split to iterate through ('train', 'val', or 'test')
        cache_dir: Directory where dataset files are cached

    Yields:
        Dictionary containing a single example from the dataset
    """
    fname = download(split, cache_dir)
    with open(fname, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def render_example(
    example: str, enc: Encoding
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, int]:
    """Process a HellaSwag example into tensors for model evaluation.

    Args:
        example: A single example from the HellaSwag dataset
        enc: Tokenizer encoding to use for tokenization

    Returns:
        Tuple containing:
            - Processed data dictionary with tokens
            - Tensor of token IDs for context and endings
            - Tensor of masks indicating which tokens are from endings
            - Integer label indicating the correct ending
    """
    ctx = example["ctx"]
    label = int(example["label"])
    endings = example["endings"]

    data: Dict[str, Any] = {"label": label, "ctx_tokens": enc.encode(ctx), "ending_tokens": []}

    tok_rows: List[List[int]] = []
    mask_rows: List[List[int]] = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(data["ctx_tokens"] + end_tokens)
        mask_rows.append([0] * len(data["ctx_tokens"]) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max([len(row) for row in tok_rows])
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)

    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


@torch.no_grad()
def evaluate(
    model: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    encoder: Encoding,
    env,
    cache_dir: str,
) -> Tuple[int, int, int]:
    """Evaluate a model on the HellaSwag validation dataset.

    Args:
        model: Model function that takes tokens and returns (logits, _)
        encoder: Tokenizer encoding to use for tokenization
        env: Training environment with device and configuration settings
        cache_dir: Directory where dataset files are cached

    Returns:
        Tuple containing:
            - Total number of examples evaluated
            - Number of correct predictions using sum loss
            - Number of correct predictions using normalized loss
    """
    if env.use_tf32:
        torch.set_float32_matmul_precision("high")

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    ddp_rank, ddp_world_size = env.ddp_config if env.ddp else (0, 1)

    for ix, example in tqdm(enumerate(iterate_examples("val", cache_dir))):
        if ix % ddp_world_size != ddp_rank:
            continue

        _, tokens, mask, label = render_example(example, encoder)
        tokens, mask = tokens.to(env.device), mask.to(env.device)

        if env.use_autocast:
            with torch.autocast(device_type=env.device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
        else:
            logits, _ = model(tokens)

        # Shift logits and tokens for next-token prediction
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()

        # Flatten for cross entropy calculation
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        # Calculate token-wise losses
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # Apply mask to only consider losses on ending tokens
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # Calculate sum and average losses for each ending
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # Get predictions (lowest loss is best)
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    # Aggregate results across distributed processes if using DDP
    if env.ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=env.device)
        num_correct = torch.tensor(num_correct, dtype=torch.long, device=env.device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=env.device)

        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)

        num_total = num_total.item()
        num_correct = num_correct.item()
        num_correct_norm = num_correct_norm.item()

    return num_total, num_correct, num_correct_norm


@app.command()
def evaluate_pretrained(
    model_type: str = typer.Option(
        "gpt2", "--model-type", "-m", help="The model type to evaluate"
    ),
    device: str = typer.Option(
        "cuda", "--device", "-d", help="Device to run evaluation on (cuda/cpu)"
    ),
    cache_dir: str = typer.Option(
        "data/.cache", "--cache-dir", "-c", help="Directory to cache dataset files"
    ),
    use_autocast: bool = typer.Option(
        False, "--autocast", help="Enable automatic mixed precision"
    ),
    use_tf32: bool = typer.Option(
        False, "--tf32", help="Enable TF32 for faster computation on GPUs"
    ),
) -> None:
    """
    Evaluate a pretrained model on the HellaSwag dataset.

    Args:
        model_type: The pretrained model type to evaluate (e.g., 'gpt2')
        device: Device to run evaluation on ('cuda' or 'cpu')
        cache_dir: Directory to cache dataset files
        use_autocast: Whether to enable automatic mixed precision
        use_tf32: Whether to enable TF32 for faster computation on GPUs
    """
    from gpt.train import TrainingEnvironment

    env = TrainingEnvironment(
        device=device,
        logger=logger.info,
        use_autocast=use_autocast,
        use_tf32=use_tf32,
        ddp_config=None,
    )

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(env.device)

    encoder = tiktoken.get_encoding(model_type)

    def model_fn(x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return model(x).logits, None

    num_total, num_correct, num_correct_norm = evaluate(model_fn, encoder, env, cache_dir)
    env.logger(
        f"Total examples: {num_total}, Accuracy: {num_correct / num_total * 100:.1f}%, "
        f"NormAccuracy: {num_correct_norm / num_total * 100:.1f}%"
    )


if __name__ == "__main__":
    app()
