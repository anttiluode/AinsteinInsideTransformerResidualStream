from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StampedResidue:
    """A branch departure plus provenance."""

    vector: Tensor
    branch: str
    role: str
    layer: int


def last_token(hidden: Tensor) -> Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"expected [batch, seq, hidden], got {tuple(hidden.shape)}")
    return hidden[:, -1, :]


@torch.inference_mode()
def hidden_at_layer(model, tokenizer, text: str, layer: int, device: str | None = None) -> Tensor:
    """Return the final-token hidden state at one transformer depth."""
    batch = tokenizer(text, return_tensors="pt")
    if device is None:
        device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    # hidden_states[0] is the embedding output; hidden_states[i+1] is layer i output.
    h = out.hidden_states[layer + 1]
    return last_token(h).detach()


@torch.inference_mode()
def final_token_hidden_trace(model, tokenizer, text: str, device: str | None = None) -> tuple[Tensor, ...]:
    """Return final-token states from embedding output through every transformer layer."""
    batch = tokenizer(text, return_tensors="pt")
    if device is None:
        device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
    return tuple(last_token(h).detach() for h in out.hidden_states)


@torch.inference_mode()
def branch_residue_trace(
    model,
    tokenizer,
    anchor_text: str,
    branch_text: str,
    device: str | None = None,
) -> tuple[Tensor, ...]:
    """Difference between branch and common-anchor final-token states at every depth."""
    anchor = final_token_hidden_trace(model, tokenizer, anchor_text, device=device)
    branch = final_token_hidden_trace(model, tokenizer, branch_text, device=device)
    if len(anchor) != len(branch):
        raise RuntimeError("anchor and branch traces have different depths")
    return tuple(b - a for a, b in zip(anchor, branch))


def branch_residue(
    model,
    tokenizer,
    anchor_text: str,
    branch_text: str,
    layer: int,
    branch: str,
    role: str,
) -> StampedResidue:
    """Contrast a branch state with the common anchor at the same layer."""
    anchor = hidden_at_layer(model, tokenizer, anchor_text, layer)
    branch_h = hidden_at_layer(model, tokenizer, branch_text, layer)
    return StampedResidue(branch_h - anchor, branch=branch, role=role, layer=layer)


class LinearComposer(nn.Module):
    """Strong attacker: output remains in the span of the two residues."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        w = torch.softmax(self.logits, dim=0)
        return w[0] * a + w[1] * b


class BilinearComposer(nn.Module):
    """
    Low-rank ordered cross-residue interaction.

    This is intentionally not a full d x d outer product. The projections create
    a compact interaction code u(a) * v(b), preserving A/B role order.
    """

    def __init__(self, hidden_size: int, rank: int = 128):
        super().__init__()
        self.a = nn.Linear(hidden_size, rank, bias=False)
        self.b = nn.Linear(hidden_size, rank, bias=False)
        self.out = nn.Linear(rank, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        z = self.a(a) * self.b(b)
        return self.norm(self.out(z))


def relative_outside_span(x: Tensor, a: Tensor, b: Tensor, eps: float = 1e-8) -> Tensor:
    """Relative norm of x after projection onto span{a,b}; batch size one expected."""
    x2 = x.reshape(-1)
    basis = torch.stack([a.reshape(-1), b.reshape(-1)], dim=1)
    coeff = torch.linalg.pinv(basis) @ x2
    residual = x2 - basis @ coeff
    return torch.linalg.vector_norm(residual) / (torch.linalg.vector_norm(x2) + eps)


def _layers(model):
    # Qwen3/Llama-family path in Hugging Face AutoModelForCausalLM.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise TypeError("unsupported model layout: expected model.model.layers")


@contextmanager
def inject_last_token(model, layer: int, delta: Tensor, scale: float = 1.0):
    """
    Add a temporary vector to the final sequence position entering layer.

    Permanent weights are untouched. This makes the synthesized residue a
    temporary lens on the resumed main stream.
    """
    block = _layers(model)[layer]

    def pre_hook(_module, args):
        if not args:
            return args
        h = args[0]
        d = delta.to(device=h.device, dtype=h.dtype)
        if d.ndim == 1:
            d = d.unsqueeze(0)
        h2 = h.clone()
        h2[:, -1, :] = h2[:, -1, :] + scale * d
        return (h2, *args[1:])

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def next_token_logits(model, tokenizer, text: str, device: str | None = None) -> Tensor:
    batch = tokenizer(text, return_tensors="pt")
    if device is None:
        device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    return model(**batch, use_cache=False, return_dict=True).logits[:, -1, :]


@torch.inference_mode()
def next_token_logits_with_injection(
    model,
    tokenizer,
    text: str,
    layer: int,
    delta: Tensor,
    scale: float = 1.0,
    device: str | None = None,
) -> Tensor:
    with inject_last_token(model, layer=layer, delta=delta, scale=scale):
        return next_token_logits(model, tokenizer, text, device=device)
