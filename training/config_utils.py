"""Small helpers for command-line config overrides and JSON dumps."""

import json
import os
from typing import Iterable

import numpy as np
import yaml


def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    """Apply ``key=value`` / ``a.b.c=value`` overrides; values are YAML-parsed.

    >>> cfg = {'optimizer': {'adam': {'lr': 1e-4}}}
    >>> apply_overrides(cfg, ['optimizer.adam.lr=3e-4', 'l2_norm=false'])['l2_norm']
    False
    """
    for item in overrides or []:
        if '=' not in item:
            raise ValueError(f'override must look like key=value, got {item!r}')
        key, raw = item.split('=', 1)
        value = yaml.safe_load(raw)
        # YAML 1.1 parses "3e-4" as a string; coerce plain numerics.
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        node = config
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return config


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_default)
