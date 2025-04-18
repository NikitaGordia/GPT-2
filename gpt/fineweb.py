import functools
import multiprocessing as mp
import os
from typing import Optional, Tuple

from datasets import Dataset, load_dataset
from loguru import logger
import numpy as np
import tiktoken
from tiktoken.core import Encoding
from tqdm import tqdm
import typer


def tokenize(doc: str, enc: Encoding, eot: int) -> np.ndarray:
    """Tokenize a document using the provided encoder.

    Args:
        doc: Document dictionary containing a 'text' field
        enc: Tokenizer encoding
        eot: End-of-text token ID

    Returns:
        Tokenized document as a numpy array of uint16 values
    """
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), (
        "token dictionary too large for uint16"
    )
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename: str, tokens_np: np.ndarray) -> None:
    """Write tokenized data to a numpy file.

    Args:
        filename: Path to save the file (without extension)
        tokens_np: Numpy array of tokens to save
    """
    try:
        np.save(filename, tokens_np)
    except IOError as e:
        logger.error(f"Error saving file {filename}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving file {filename}: {e}")
        raise


class ShardBuffer:
    """Class for managing shards of tokenized data.

    This class implements the context manager protocol, so it can be used with
    the 'with' statement. The finalize() method will be called automatically
    when exiting the context.
    """

    def __init__(self, local_dir: str, shard_size: int):
        """Initialize the shard buffer.

        Args:
            local_dir: Directory to save processed files
            shard_size: Size of each shard in tokens
        """
        self.local_dir = local_dir
        self.shard_size = shard_size
        self.shard_index = 0
        self.token_count = 0
        self.all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
        self.progress_bar: Optional[tqdm] = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.finalize()

    def initialize_shard(self) -> None:
        """Initialize a new shard with empty array."""
        self.all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
        self.token_count = 0
        self.progress_bar = None

    def save_shard(self) -> None:
        """Save a shard of tokens."""
        split = "val" if self.shard_index == 0 else "train"
        filename = os.path.join(self.local_dir, f"edufineweb_{split}_{self.shard_index:06d}")
        write_datafile(filename, self.all_tokens_np[: self.token_count])
        logger.info(f"Saved shard {self.shard_index} to {filename}")

    def add_tokens(self, tokens: np.ndarray) -> None:
        """Add tokens to the current shard and handle shard transitions.

        Args:
            tokens: Tokens to add to the current shard
        """
        if self.token_count + len(tokens) < self.shard_size:
            # Add tokens to current shard
            self.all_tokens_np[self.token_count : self.token_count + len(tokens)] = tokens
            self.token_count += len(tokens)
            if self.progress_bar is None:
                self.progress_bar = tqdm(
                    total=self.shard_size, unit="tokens", desc=f"Shard {self.shard_index}"
                )
            self.progress_bar.update(len(tokens))
        else:
            # Shard is full, save it and start a new one
            remainder = self.shard_size - self.token_count
            if self.progress_bar is not None:
                self.progress_bar.update(remainder)
                self.progress_bar.close()

            self.all_tokens_np[self.token_count : self.token_count + remainder] = tokens[
                :remainder
            ]
            self.save_shard()
            self.shard_index += 1

            # Start new shard with remaining tokens
            new_remainder = len(tokens) - remainder
            self.initialize_shard()
            if new_remainder > 0:
                self.all_tokens_np[0:new_remainder] = tokens[remainder:]
                self.token_count = new_remainder

    def finalize(self) -> None:
        """Save any remaining tokens and clean up resources."""
        if self.token_count > 0:
            assert self.token_count < self.shard_size, (
                f"Token count ({self.token_count}) must be less than shard size ({self.shard_size})"
            )
            self.save_shard()
            if self.progress_bar is not None:
                self.progress_bar.close()


class FinewebProcessor:
    """Class for downloading and processing the Fineweb dataset."""

    def __init__(self, local_dir: str, remote_name: str, shard_size: int):
        """Initialize the Fineweb processor.

        Args:
            local_dir: Directory to save processed files
            remote_name: Remote dataset name
            shard_size: Size of each shard in tokens
        """
        self.local_dir = local_dir
        self.remote_name = remote_name
        self.shard_size = shard_size

    def setup_directories(self) -> None:
        """Create output directories if they don't exist."""
        logger.info(f"Setting up output directory: {self.local_dir}")
        os.makedirs(self.local_dir, exist_ok=True)

    def load_dataset(self) -> Dataset:
        """Load the Fineweb dataset.

        Returns:
            The loaded dataset
        """
        logger.info(f"Loading dataset {self.remote_name}")
        try:
            return load_dataset("HuggingFaceFW/fineweb-edu", name=self.remote_name, split="train")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def setup_tokenizer(self) -> Tuple[Encoding, int]:
        """Set up the tokenizer.

        Returns:
            Tuple of (encoding, end-of-text token)
        """
        logger.info("Setting up tokenizer")
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens["<|endoftext|>"]
        return enc, eot

    def process(self) -> None:
        """Process the Fineweb dataset."""
        self.setup_directories()
        dataset = self.load_dataset()
        enc, eot = self.setup_tokenizer()

        # Create a partial function with the tokenizer arguments
        tokenize_with_args = functools.partial(tokenize, enc=enc, eot=eot)

        # Set up multiprocessing
        nprocs = max(1, os.cpu_count() // 2)
        logger.info(f"Using {nprocs} processes for tokenization")

        try:
            with mp.Pool(nprocs) as pool, ShardBuffer(self.local_dir, self.shard_size) as buffer:
                for tokens in pool.imap(tokenize_with_args, dataset, chunksize=32):
                    buffer.add_tokens(tokens)

            logger.success("Done processing Fineweb dataset")
        except Exception as e:
            logger.error(f"Error processing dataset: {e}")
            raise


def download_and_process(
    local_dir: str = typer.Argument(..., help="Directory to save processed files"),
    remote_name: str = typer.Option("sample-10BT", help="Remote dataset name"),
    shard_size: int = typer.Option(int(1e8), help="Size of each shard"),
) -> None:
    """Download and process the Fineweb dataset.

    Args:
        local_dir: Directory to save processed files
        remote_name: Remote dataset name
        shard_size: Size of each shard in tokens
    """
    processor = FinewebProcessor(local_dir, remote_name, shard_size)
    processor.process()


if __name__ == "__main__":
    typer.run(download_and_process)
