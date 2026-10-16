"""A run's record: everything ER was sent and answered, every tool call, every
picture, in one place.

    logs/crabgym/<timestamp>/
    ├── run.json            the instruction, the agent and its model, the seed, the spawn, the process
    ├── conversation.jsonl  ER-led: everything ER was sent and answered, in order (`agent/conversation.py`)
    ├── skills.jsonl        one line per tool call: arguments, result, sim time
    ├── truth.jsonl         ER-led: the evaluator's truth after each call
    ├── calls.jsonl         executor-led: one line per planner call (ER-led runs before 2026-10-10 too)
    ├── images/             every picture and clip handed over, as it was handed
    └── report.json         the evaluator's: counts, errors, what really happened

Three uses, and the second is why the images are kept as sent rather than
regenerated: the evaluator scores a run from it, a replay (`--agent
replay:<dir>`) drives the tools through the calls a logged run made, without
the API -- a demonstration that costs nothing and changes nothing -- and the
console (`python -m crabgym.web`) shows it, while it is written and after.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

#: Under the repository's `logs/`, beside training's `logs/<model>/<task>/`.
LOG_ROOT = Path(__file__).resolve().parents[1] / "logs" / "crabgym"
#: The ER-led run's record of the conversation, one JSON object a line, appended
#: as the run goes; `agent/conversation.py` has its form.
CONVERSATION = "conversation.jsonl"


def _plain(value: Any) -> Any:
    """What JSON can carry, from what a dataclass or numpy hands over."""
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class RunLog:
    """One run's directory, opened for writing."""

    root: Path

    @classmethod
    def open(cls, where: Path | None = None) -> RunLog:
        if where is not None:
            where.mkdir(parents=True, exist_ok=False)
            (where / "images").mkdir()
            return cls(where)
        stamp = f"{datetime.now().astimezone():%Y-%m-%d_%H-%M-%S}"
        # Two runs started in the same second get distinct records: the
        # directory's creation is the claim, so there is no window between
        # looking and taking.
        for n in range(1, 100):
            root = LOG_ROOT / (stamp if n == 1 else f"{stamp}-{n}")
            try:
                root.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            (root / "images").mkdir()
            return cls(root)
        raise FileExistsError(f"{LOG_ROOT / stamp}: a hundred runs in one second")

    def write(self, name: str, record: Any) -> None:
        (self.root / name).write_text(
            json.dumps(_plain(record), indent=2, ensure_ascii=False) + "\n", "utf-8")

    def append(self, name: str, record: Any) -> None:
        with (self.root / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_plain(record), ensure_ascii=False) + "\n")

    def image(self, name: str, frame: np.ndarray) -> Path:
        """Keep a frame, PNG, under `images/`; returns its path."""
        from PIL import Image

        path = self.root / "images" / f"{name}.png"
        Image.fromarray(frame).save(path)
        return path

    def skill(self, name: str, t: float, args: dict[str, Any], result: Any) -> None:
        self.append("skills.jsonl", {"skill": name, "t": t, "args": args, "result": result,
                                     "wall": time.time()})

    def call(self, record: dict[str, Any]) -> None:
        self.append("calls.jsonl", {**record, "wall": time.time()})

    def event(self, kind: str, **fields: Any) -> None:
        """One line of the conversation's record (`CONVERSATION`)."""
        self.append(CONVERSATION, {"kind": kind, **fields, "wall": time.time()})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every record in a `.jsonl` file this module wrote.

    Split at "\\n" alone: `str.splitlines` also splits at U+2028, U+2029 and
    U+0085, which `json.dumps(ensure_ascii=False)` leaves as they are inside a
    string -- one in ER's words would cut its line in two."""
    text = path.read_text("utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


__all__ = ["CONVERSATION", "LOG_ROOT", "RunLog", "read_jsonl"]
