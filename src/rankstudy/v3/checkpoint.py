"""Adapter-only safe checkpoints with optimizer, allocation and RNG state."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .adapters import adapted


def json_write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def backbone_hash(model):
    """Hash original frozen tensors; no adapter tensors or copies on disk."""
    h = hashlib.sha256()
    bases = tuple(name + ".base." for name in adapted(model))
    for name, p in model.named_parameters():
        if p.requires_grad:
            continue
        h.update(name.replace(".base.", ".").encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, model, optimizer, extra):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    tensors, description = {}, {}
    for name, m in adapted(model).items():
        description[name] = {"rank": m.rank, "init": m.init, "score": m.score,
                             "scale": float(m.scale), "slot_ids": m.slot_ids.tolist()}
        for key, value in m.state_dict().items():
            if not key.startswith("base."):
                tensors[f"{name}.{key}"] = value.detach().cpu().contiguous().clone()
    save_file(tensors, str(path / "adapter.safetensors"))
    state = {"optimizer": optimizer.state_dict(), "extra": extra,
             "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    # Only tensors and primitive containers. Loading always uses weights_only.
    torch.save(state, path / "training_state.pt")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [path / "adapter.safetensors", path / "training_state.pt"]}
    json_write(path / "manifest.json", {"schema": "rankstudy-v3-checkpoint", "modules": description,
                                         "files": files, "complete": True})


def load_checkpoint(path, model, optimizer, device):
    path = Path(path)
    desc = json.loads((path / "manifest.json").read_text())
    for name, digest in desc['files'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"checkpoint hash mismatch: {name}")
    tensors = load_file(str(path / "adapter.safetensors"), device=str(device))
    for name, m in adapted(model).items():
        got = {key[len(name)+1:]: val for key, val in tensors.items() if key.startswith(name+'.')}
        expected = {key for key in m.state_dict() if not key.startswith('base.')}
        if set(got) != expected:
            raise ValueError(f"incomplete adapter checkpoint: {name}")
        result = m.load_state_dict(got, strict=False)
        if result.unexpected_keys or any(not key.startswith('base.') for key in result.missing_keys):
            raise ValueError(str(result))
    state = torch.load(path/'training_state.pt', map_location=device, weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'].cpu())
    random.setstate(state['python_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
    return state['extra']
