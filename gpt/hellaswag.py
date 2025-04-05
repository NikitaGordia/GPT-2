import json
import os

import requests
import tiktoken
import torch
import torch.distributed as dist
from torch.nn import functional as F
from tqdm import tqdm
from transformers import GPT2LMHeadModel
import typer

from utils import setup_typer

HELLASWAGS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

app = setup_typer("hellaswag")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
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


def download(split, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    data_url = HELLASWAGS[split]

    data_filename = os.path.join(cache_dir, f"hellaswag_{split}.json")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    return data_filename


def iterate_examples(split, cache_dir):
    fname = download(split, cache_dir)
    with open(fname, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def render_example(example, enc):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {"label": label, "ctx_tokens": enc.encode(ctx), "ending_tokens": []}

    tok_rows = []
    mask_rows = []
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
def evaluate(model, encoder, env, cache_dir):
    if env.use_tf32:
        torch.set_float32_matmul_precision("high")

    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    ddp_rank, ddp_world_size = env.ddp_config if env.ddp else (0, 1)

    for ix, example in tqdm(enumerate(iterate_examples("val", cache_dir))):
        if ix % ddp_world_size != ddp_rank:
            continue

        data, tokens, mask, label = render_example(example, encoder)
        tokens, mask = tokens.to(env.device), mask.to(env.device)

        if env.use_autocast:
            with torch.autocast(device_type=env.device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
        else:
            logits, _ = model(tokens)

        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()

        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction="none")
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

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
):
    """
    Evaluate a pretrained model on the HellaSwag dataset.
    """
    from gpt.train import TrainingEnvironment

    env = TrainingEnvironment(
        device=device,
        logger=print,
        use_autocast=use_autocast,
        use_tf32=use_tf32,
        ddp_config=None,
    )

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(env.device)

    encoder = tiktoken.get_encoding(model_type)

    def model_fn(x):
        return model(x).logits, None

    num_total, num_correct, num_correct_norm = evaluate(model_fn, encoder, env, cache_dir)
    env.logger(
        f"Total examples: {num_total}, Accuracy: {num_correct / num_total * 100:.1f}%, NormAccuracy: {num_correct_norm / num_total * 100:.1f}.%"
    )


if __name__ == "__main__":
    app()
