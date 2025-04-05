import functools
import multiprocessing as mp
import os

from datasets import load_dataset
from loguru import logger
import numpy as np
import tiktoken
from tqdm import tqdm
import typer


def tokenize(doc, enc, eot):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def download_and_process(
    local_dir: str = typer.Argument(..., help="Directory to save processed files"),
    remote_name: str = typer.Option("sample-10BT", help="Remote dataset name"),
    shard_size: int = typer.Option(int(1e8), help="Size of each shard"),
):
    """Process and download Fineweb dataset."""
    logger.info(f"Starting download and processing to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)

    logger.info(f"Loading dataset {remote_name}")
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    tokenize_with_args = functools.partial(tokenize, enc=enc, eot=eot)

    nprocs = max(1, os.cpu_count() // 2)
    logger.info(f"Using {nprocs} processes for tokenization")

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize_with_args, fw, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count : token_count + len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(
                        total=shard_size, unit="tokens", desc=f"Shard {shard_index}"
                    )
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")

                remainer = shard_size - token_count
                progress_bar.update(remainer)
                all_tokens_np[token_count : token_count + remainer] = tokens[:remainer]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None

                new_remainder = len(tokens) - remainer
                all_tokens_np[0:new_remainder] = tokens[remainer:]
                token_count = new_remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(local_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])

    logger.success("Done processing Fineweb dataset")


if __name__ == "__main__":
    typer.run(download_and_process)
