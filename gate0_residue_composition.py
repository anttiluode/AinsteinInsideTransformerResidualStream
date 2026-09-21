from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from latent_fork import hidden_at_layer, relative_outside_span


MOD = 7
ADDS = tuple(range(MOD))
MULS = tuple(range(1, MOD))


def mapping_prompt(mapping: tuple[int, ...]) -> str:
    rows = "\n".join(f"{x} -> {mapping[x]}" for x in range(MOD))
    return (
        "A machine maps one digit to one digit. Study the complete table.\n"
        f"{rows}\n"
        "Internal rule representation:"
    )


def identity_mapping() -> tuple[int, ...]:
    return tuple(range(MOD))


def add_mapping(a: int) -> tuple[int, ...]:
    return tuple((x + a) % MOD for x in range(MOD))


def mul_mapping(b: int) -> tuple[int, ...]:
    return tuple((b * x) % MOD for x in range(MOD))


def composed_mapping(a: int, b: int) -> tuple[int, ...]:
    return tuple((b * ((x + a) % MOD)) % MOD for x in range(MOD))


@dataclass
class PairExample:
    a: int
    b: int
    ra: Tensor
    rb: Tensor
    target: Tensor


def pair_split(a: int, b: int) -> str:
    # Pair-level split. Every individual add and multiply operation still appears in train.
    code = (3 * a + 5 * b + a * b) % 7
    if code == 0:
        return "test"
    if code == 1:
        return "val"
    return "train"


class LinearConcat(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Linear(2 * d, d, bias=False)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat([a, b], dim=-1))


class Bilinear(nn.Module):
    def __init__(self, d: int, rank: int):
        super().__init__()
        self.a = nn.Linear(d, rank, bias=False)
        self.b = nn.Linear(d, rank, bias=False)
        self.out = nn.Linear(rank, d, bias=False)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.out(self.a(a) * self.b(b))


class MLPConcat(nn.Module):
    def __init__(self, d: int, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, width),
            nn.SiLU(),
            nn.Linear(width, d),
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat([a, b], dim=-1))


def r2_score(pred: Tensor, target: Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean(dim=0, keepdim=True)) ** 2)
    return (1.0 - ss_res / ss_tot.clamp_min(1e-12)).item()


def cosine_mean(pred: Tensor, target: Tensor) -> float:
    return torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()


def fit_model(
    model: nn.Module,
    train: list[PairExample],
    val: list[PairExample],
    steps: int,
    lr: float,
    device: str,
) -> nn.Module:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best = None
    best_val = math.inf

    def stack(examples):
        a = torch.cat([e.ra for e in examples], dim=0).to(device)
        b = torch.cat([e.rb for e in examples], dim=0).to(device)
        y = torch.cat([e.target for e in examples], dim=0).to(device)
        return a, b, y

    ta, tb, ty = stack(train)
    va, vb, vy = stack(val)

    for step in range(steps):
        model.train()
        pred = model(ta, tb)
        loss = torch.nn.functional.mse_loss(pred, ty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                v = torch.nn.functional.mse_loss(model(va, vb), vy).item()
            if v < best_val:
                best_val = v
                best = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}

    assert best is not None
    model.load_state_dict(best)
    return model


def evaluate(model: nn.Module, examples: list[PairExample], device: str) -> dict:
    model.eval()
    a = torch.cat([e.ra for e in examples], dim=0).to(device)
    b = torch.cat([e.rb for e in examples], dim=0).to(device)
    y = torch.cat([e.target for e in examples], dim=0).to(device)
    with torch.no_grad():
        p = model(a, b)
    outside = [
        relative_outside_span(p[i:i+1], a[i:i+1], b[i:i+1]).item()
        for i in range(p.shape[0])
    ]
    return {
        "r2": r2_score(p, y),
        "cosine": cosine_mean(p, y),
        "outside_branch_span_mean": float(sum(outside) / len(outside)),
    }


def branch_baselines(examples: list[PairExample]) -> dict:
    a = torch.cat([e.ra for e in examples], dim=0)
    b = torch.cat([e.rb for e in examples], dim=0)
    y = torch.cat([e.target for e in examples], dim=0)

    # Oracle scalar convex blend on the held-out split: intentionally generous attacker.
    grid = torch.linspace(0, 1, 101)
    best_r2 = -float("inf")
    best_w = None
    for w in grid:
        p = w * a + (1 - w) * b
        r = r2_score(p, y)
        if r > best_r2:
            best_r2, best_w = r, float(w)

    return {
        "branch_a_r2": r2_score(a, y),
        "branch_b_r2": r2_score(b, y),
        "oracle_convex_r2": best_r2,
        "oracle_convex_weight_a": best_w,
    }


@torch.inference_mode()
def collect(model, tok, layer: int, device: str | None) -> list[PairExample]:
    anchor = hidden_at_layer(model, tok, mapping_prompt(identity_mapping()), layer, device=device)

    add_res = {}
    for a in ADDS:
        h = hidden_at_layer(model, tok, mapping_prompt(add_mapping(a)), layer, device=device)
        add_res[a] = (h - anchor).float().cpu()

    mul_res = {}
    for b in MULS:
        h = hidden_at_layer(model, tok, mapping_prompt(mul_mapping(b)), layer, device=device)
        mul_res[b] = (h - anchor).float().cpu()

    examples = []
    for a in ADDS:
        for b in MULS:
            h = hidden_at_layer(model, tok, mapping_prompt(composed_mapping(a, b)), layer, device=device)
            target = (h - anchor).float().cpu()
            examples.append(PairExample(a=a, b=b, ra=add_res[a], rb=mul_res[b], target=target))
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--layer", type=int, default=18)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--mlp-width", type=int, default=128)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results/gate0_residue_composition.json")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
    ).eval()
    for param in base.parameters():
        param.requires_grad_(False)

    examples = collect(base, tok, args.layer, device=None)
    splits = {
        name: [e for e in examples if pair_split(e.a, e.b) == name]
        for name in ("train", "val", "test")
    }
    if not all(splits.values()):
        raise RuntimeError({k: len(v) for k, v in splits.items()})

    d = examples[0].target.shape[-1]
    models = {
        "linear_concat": LinearConcat(d),
        "bilinear": Bilinear(d, args.rank),
        "mlp_concat": MLPConcat(d, args.mlp_width),
    }

    receipt = {
        "model": args.model,
        "layer": args.layer,
        "hidden_size": d,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "branch_baselines": branch_baselines(splits["test"]),
        "models": {},
    }

    for name, model in models.items():
        fit = fit_model(
            model,
            splits["train"],
            splits["val"],
            steps=args.steps,
            lr=args.lr,
            device=args.train_device,
        )
        metrics = evaluate(fit, splits["test"], args.train_device)
        metrics["parameters"] = sum(p.numel() for p in fit.parameters())
        receipt["models"][name] = metrics

        if name == "bilinear":
            fit.eval()
            a = torch.cat([e.ra for e in splits["test"]], dim=0).to(args.train_device)
            b = torch.cat([e.rb for e in splits["test"]], dim=0).to(args.train_device)
            y = torch.cat([e.target for e in splits["test"]], dim=0).to(args.train_device)
            with torch.no_grad():
                swapped = fit(b, a)
            receipt["models"][name]["swapped_role_r2"] = r2_score(swapped, y)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print("\nThis is Gate 0a representation composition only.")
    print("It is not yet a win over extra-token reasoning; Gate 0b must inject and compare compute.")


if __name__ == "__main__":
    main()
