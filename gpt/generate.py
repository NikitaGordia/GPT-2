import hydra
from omegaconf import DictConfig
import tiktoken
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from gpt.model import GPT, GPTConfig
from gpt.utils import RuntimeEnvironment


def generate(
    model: nn.Module,
    env: RuntimeEnvironment,
    text: str,
    enc: tiktoken.Encoding,
    num_return_sequences: int = 5,
    max_length: int = 30,
) -> None:
    model.eval()
    model.to(env.device)

    tokens = enc.encode(text)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(env.device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    for _ in tqdm(range(max_length - x.size(1))):
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_from_pretrained(cfg: DictConfig) -> None:
    env = RuntimeEnvironment.from_hydra_config(cfg.env)

    gpt_config = GPTConfig.from_hydra_config(cfg.model)
    model = GPT.from_pretrained(gpt_config)
    enc = tiktoken.get_encoding(cfg.tokens.encoding)

    generate(
        model,
        env,
        cfg.gen.prompt,
        enc,
        cfg.gen.parallel_sequences,
        cfg.gen.max_length,
    )


if __name__ == "__main__":
    generate_from_pretrained()
