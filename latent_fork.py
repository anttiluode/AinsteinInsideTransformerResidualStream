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


def _extract_hidden(output):
    """Extract hidden states from a decoder-block forward-hook output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    raise TypeError(f"unsupported decoder-block output type: {type(output)!r}")


@torch.inference_mode()
def final_token_block_trace(
    model,
    tokenizer,
    text: str,
    device: str | None = None,
) -> tuple[Tensor, ...]:
    """
    Return raw final-token outputs from every decoder block, before final norm.

    Hugging Face's output_hidden_states convention appends the model's final
    normalized state in the last slot for Llama/Qwen-style decoders. That makes
    the last slot incomparable to earlier raw block outputs. Forward hooks on the
    decoder blocks avoid that boundary artifact.
    """
    layers = _layers(model)
    captured: list[Tensor | None] = [None] * len(layers)
    handles = []

    for index, block in enumerate(layers):
        def hook(_module, _args, output, index=index):
            h = _extract_hidden(output)
            captured[index] = last_token(h).detach().float().cpu()

        handles.append(block.register_forward_hook(hook))

    try:
        batch = tokenizer(text, return_tensors="pt")
        if device is None:
            device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        model(**batch, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()

    if any(h is None for h in captured):
        missing = [i for i, h in enumerate(captured) if h is None]
        raise RuntimeError(f"failed to capture decoder layers {missing}")
    return tuple(h for h in captured if h is not None)


@torch.inference_mode()
def hidden_at_layer(model, tokenizer, text: str, layer: int, device: str | None = None) -> Tensor:
    """Return the raw post-block final-token hidden state at decoder layer index."""
    trace = final_token_block_trace(model, tokenizer, text, device=device)
    if not 0 <= layer < len(trace):
        raise ValueError(f"layer {layer} outside 0..{len(trace)-1}")
    return trace[layer]


@torch.inference_mode()
def branch_residue_trace(
    model,
    tokenizer,
    anchor_text: str,
    branch_text: str,
    device: str | None = None,
) -> tuple[Tensor, ...]:
    """Difference between branch and common-anchor raw block outputs at every depth."""
    anchor = final_token_block_trace(model, tokenizer, anchor_text, device=device)
    branch = final_token_block_trace(model, tokenizer, branch_text, device=device)
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
def inject_token_position(
    model,
    layer: int,
    delta: Tensor,
    *,
    position: int = -1,
    scale: float = 1.0,
    once: bool = False,
):
    """
    Add a temporary vector at one token position entering a decoder block.

    position=-1 means the final token. With once=True, only the first
    matching forward call is modified. The latter is useful during generation:
    edit the prompt/workspace state once, then let cached causal computation
    unfold without repeatedly steering every newly generated token.

    If the selected position is outside a later one-token decode call, the hook
    simply does nothing.
    """
    block = _layers(model)[layer]
    fired = False

    def pre_hook(_module, args):
        nonlocal fired
        if not args or (once and fired):
            return args

        h = args[0]
        pos = position if position >= 0 else h.shape[1] + position
        if pos < 0 or pos >= h.shape[1]:
            return args

        d = delta.to(device=h.device, dtype=h.dtype)
        if d.ndim == 1:
            d = d.unsqueeze(0)

        h2 = h.clone()
        h2[:, pos, :] = h2[:, pos, :] + scale * d
        fired = True
        return (h2, *args[1:])

    handle = block.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def inject_last_token(
    model,
    layer: int,
    delta: Tensor,
    scale: float = 1.0,
    *,
    once: bool = False,
):
    """Backward-compatible final-token wrapper around inject_token_position."""
    with inject_token_position(
        model,
        layer,
        delta,
        position=-1,
        scale=scale,
        once=once,
    ):
        yield


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
