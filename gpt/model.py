from dataclasses import dataclass
import inspect
from typing import Any, Dict, List, Optional, Tuple

import hydra
from loguru import logger
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration class for GPT model parameters."""

    name: str = "gpt2"

    block_size: int = 1024  # Maximum sequence length
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layer: int = 12  # Number of transformer layers
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimension

    @classmethod
    def from_hydra_config(cls, cfg: DictConfig) -> "GPTConfig":
        return hydra.utils.instantiate(cfg)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer with scaled dot product attention."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gelu(self.c_fc(x))
        return self.c_proj(x)


class Block(nn.Module):
    """Transformer block with self-attention and MLP."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT-2 language model implementation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # position embeddings
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # transformer blocks
                ln_f=nn.LayerNorm(config.n_embd),  # final layer norm
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Block 1: Setup inputs
        _, seq_len = ids.size()
        assert seq_len <= self.config.block_size, (
            f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
        )

        # Block 2: Embedding lookup and positional encoding
        value_embedding = self.transformer.wte(ids)
        position = torch.arange(0, seq_len, dtype=torch.long, device=ids.device)
        pos_embedding = self.transformer.wpe(position)
        x = value_embedding + pos_embedding

        # Block 3: Transformer blocks processing
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # Block 4: Logits calculation
        logits = self.lm_head(x)

        # Block 5: Loss calculation (if targets provided)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Block 6: Return outputs
        return logits, loss

    @classmethod
    def from_pretrained(cls, gpt_cnf: GPTConfig) -> "GPT":
        """Load a pre-trained GPT-2 model from HuggingFace.

        Args:
            model_type: One of 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
            cfg: Optional Hydra configuration

        Returns:
            A GPT model with weights loaded from the pre-trained model
        """
        from transformers import GPT2LMHeadModel

        logger.info(f"Loading weights from pretrained GPT: {gpt_cnf.name}")
        model = GPT(gpt_cnf)

        # Load state dictionaries
        sd: Dict[str, torch.Tensor] = model.state_dict()
        sd_keys: List[str] = [k for k in sd.keys() if not k.endswith(".attn.bias")]
        model_hf = GPT2LMHeadModel.from_pretrained(gpt_cnf.name)
        sd_hf: Dict[str, torch.Tensor] = model_hf.state_dict()
        sd_keys_hf: List[str] = [k for k in sd_hf.keys() if not k.endswith(".attn.masked_bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]

        # Weights that need to be transposed
        transposed: List[str] = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        assert len(sd_keys_hf) == len(sd_keys), (
            f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        )

        # Copy weights from HuggingFace model to our model
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        device_type: str,
    ) -> torch.optim.AdamW:
        """Configure the optimizer with weight decay applied only to 2D+ parameters."""
        param_dict: Dict[str, torch.nn.Parameter] = {
            pn: p for pn, p in self.named_parameters() if p.requires_grad
        }
        decay_params: List[torch.nn.Parameter] = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params: List[torch.nn.Parameter] = [
            p for _, p in param_dict.items() if p.dim() < 2
        ]
        optim_groups: List[Dict[str, Any]] = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available: bool = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused: bool = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
