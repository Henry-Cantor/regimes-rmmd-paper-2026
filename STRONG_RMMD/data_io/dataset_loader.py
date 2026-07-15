from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


SHARDED_FORMAT = "phase0-sharded-v1"


def load_phase0_payload(path: Path) -> Dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid Phase-0 payload: {path}")
    return payload


def is_sharded_payload(payload: Dict) -> bool:
    return payload.get("format") == SHARDED_FORMAT and "shards" in payload


class Phase0DatasetView:
    def __init__(self, payload: Dict):
        self.payload = payload
        self.sharded = is_sharded_payload(payload)
        self.samples = payload.get("data", payload.get("samples", [])) if not self.sharded else []
        self.shards = [Path(p) for p in payload.get("shards", [])] if self.sharded else []
        self.sample_index = payload.get("sample_index", []) if self.sharded else []
        self._shard_cache_idx = None
        self._shard_cache_payload = None

    def __len__(self) -> int:
        return len(self.sample_index) if self.sharded else len(self.samples)

    def _load_shard(self, shard_idx: int) -> Dict:
        if self._shard_cache_idx != shard_idx:
            self._shard_cache_payload = torch.load(self.shards[shard_idx], map_location="cpu", weights_only=False)
            self._shard_cache_idx = shard_idx
        return self._shard_cache_payload

    def get_sample(self, index: int) -> Dict:
        if self.sharded:
            entry = self.sample_index[index]
            shard_payload = self._load_shard(int(entry["shard_idx"]))
            return shard_payload["data"][int(entry["sample_idx"])]
        return self.samples[index]

    def iter_samples(self):
        for index in range(len(self)):
            yield self.get_sample(index)
