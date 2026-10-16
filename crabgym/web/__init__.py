"""The console: every run's conversation with ER, live while it is written.

    python -m crabgym.web            # then http://127.0.0.1:8765/

`record` reads the runs' records under `logs/crabgym/` -- the conversation an
ER-led run writes (`crabgym/agent/conversation.py` has its form), and the
records of runs from before it, turned into the same events -- and `server`
serves them with the page in `static/`. Read-only; and nothing here imports
`crabgym.agent`, which would bring MuJoCo in for a page that reads JSON.
"""

#: The console's port, the same on every start so that a bookmark keeps working.
PORT = 8765

__all__ = ["PORT"]
