"""`python -m crabgym "<instruction>"`: one run of the gym.

    python -m crabgym "做 3 个引体向上" --agent gemini            # Gemini Robotics-ER 2
    python -m crabgym "做 3 个引体向上" --agent gemini --viewer   # ... watched live (needs a display)
    python -m crabgym "做 3 个引体向上" --agent gemini --seed 3   # ... set down elsewhere on the deck
    python -m crabgym "做 3 个引体向上" --agent gemini --thoughts # ... with its thought summaries
    python -m crabgym "做 3 个引体向上"                          # the ER emulator: no key, no model
    python -m crabgym "做 3 个引体" --agent replay:logs/crabgym/2026-10-07_12-00-00

The world is the pull-up station's map under the gym bundle; the robot is set
down somewhere on the deck facing anywhere (`--seed`), and ER is given the
instruction. The run is ER-led (`crabgym/agent`): ER calls every step through
the tools, and actions run while it waits on them. Gemini needs
`GOOGLE_API_KEY`, in the environment or in the repository's `.env.local`
(git-ignored). The run's record lands in `logs/crabgym/<timestamp>/` and the
verdict is printed at the end; `python -m crabgym.web` shows the conversation
with ER while it goes on, and every run's since.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .log import LOG_ROOT
from .web import PORT

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m crabgym", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instruction", help="what ER is told, in any language")
    parser.add_argument("--agent", default="emulator", help="ER-led: emulator, gemini, or replay:<run dir>")
    parser.add_argument("--realtime", action="store_true",
                        help="ER-led: let the running action go on while ER thinks, as on the robot")
    parser.add_argument("--thoughts", action="store_true",
                        help=("--agent gemini: ask for ER's thought summaries, to read in the record; they go back "
                              "to ER with its answers, so every later request carries them"))
    parser.add_argument("--seed", type=int, default=0, help="where the robot is set down, and facing where")
    parser.add_argument("--app", type=Path, default=REPO_ROOT / "out" / "bundle_gym" / "gym.app",
                        help="the gym bundle's .app (scripts/deploy.py --manifest gym)")
    parser.add_argument("--scene", type=Path, default=None, help="a .map other than the station's own")
    parser.add_argument("--look", choices=("hall", "plain"), default="hall",
                        help="hall: the station in a gym hall (crabgym/hall.py); plain: the map as it is")
    parser.add_argument("--viewer", action="store_true", help="watch it live (needs a display)")
    parser.add_argument("--unbound", action="store_true",
                        help="load the map even if its digest is not the station's (a map being edited)")
    parser.add_argument("--max-calls", type=int, default=None, help="the most turns ER is given (default 200)")
    parser.add_argument("--log-dir", type=Path, default=None, help="where to write the record (default logs/crabgym/<timestamp>)")
    args = parser.parse_args(argv)
    if args.thoughts and args.agent != "gemini":
        parser.error("--thoughts asks Gemini for its thought summaries: it needs --agent gemini")
    from .agent.loop import AgentRun

    run = AgentRun.build(args.instruction, agent=args.agent, seed=args.seed, app=args.app, scene=args.scene,
                         viewer=args.viewer, bind=not args.unbound, log_dir=args.log_dir, look=args.look,
                         realtime=args.realtime, thoughts=args.thoughts)
    _watch(run.log.root)
    try:
        verdict = run.go(max_turns=args.max_calls or 200)
    finally:
        run.close()
    shown = {k: v for k, v in verdict.items() if k not in ("skills",)}
    if isinstance(shown.get("locate_bar"), dict):
        shown["locate_bar"] = {k: v for k, v in shown["locate_bar"].items() if k != "each"}
    print(json.dumps(shown, indent=2, ensure_ascii=False))
    print(f"[run] record: {run.log.root}")
    return 0 if verdict.get("ok") else 1


def _watch(record: Path) -> None:
    """Where the record is, and how to watch it being written."""
    root = record.resolve().parent
    console = "python -m crabgym.web" + ("" if root == LOG_ROOT else f" --root {root}")
    print(f"[run] record: {record} -- watch it: {console}, then http://127.0.0.1:{PORT}/#/{record.name}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
