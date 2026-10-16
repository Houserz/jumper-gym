"""The guard: checks that run before an operation and may refuse it.

The mechanism is kept and the limits are not set. A check is a function of
the operation's name, its arguments and the world model that returns None to
let it through or a refusal -- a reason and a suggestion -- that goes back to
ER as the operation's result, without anything moving. `Guard()` holds no
checks at all, so it lets everything through; which checks a run has, and at
what thresholds, is a decision taken separately and passed in.

A refusal never changes what ER asked for into something else: ER reads why,
and decides.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

#: `check(operation, args, world) -> None | {"reason": ..., "suggestion": ...}`.
Check = Callable[[str, dict[str, Any], Any], dict[str, str] | None]


@dataclass
class Guard:
    #: The checks, by operation name; `"*"` runs before every operation.
    checks: dict[str, list[Check]] = field(default_factory=dict)

    def add(self, operation: str, check: Check) -> None:
        self.checks.setdefault(operation, []).append(check)

    def review(self, operation: str, args: dict[str, Any], world: Any) -> dict[str, Any] | None:
        """None when the operation may go; otherwise the refusal ER gets."""
        for check in (*self.checks.get("*", ()), *self.checks.get(operation, ())):
            refusal = check(operation, args, world)
            if refusal:
                return {"ok": False, "error": f"{operation} refused by the guard", **refusal}
        return None


__all__ = ["Check", "Guard"]
