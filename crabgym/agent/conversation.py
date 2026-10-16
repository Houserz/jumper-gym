"""The conversation with ER in the loop's own terms, and the record of it.

Every message ER is sent is rendered here, once, from what the loop has --
the instruction, the tool results, the situation report, the pictures and
clips the results named, the reminder -- into **parts** that no model's API
has a say in. The loop writes those parts into the run's record before ER is
asked (`log.CONVERSATION`, `logs/crabgym/<run>/conversation.jsonl`), and an
agent that talks to a model translates the same parts into its API and sends
nothing else. So the record is what ER was sent, and another model needs only
a translation: `GeminiAgent._to_genai` is the one there is, and
`tests/test_crabgym_agent.py` rebuilds every request Gemini got from the
record and holds it against what the client received.

## The record

One JSON object a line, each with `wall`, the time it was written. `AgentRun`
writes them all:

    {"kind": "setup", "agent", ...}            once, first: the agent's `setup`
    {"kind": "message", "role": "user", "turn", "t", "parts", "nudge"?}
    {"kind": "message", "role": "model", "turn", "parts", "latency_s",
     "finish"?, "tokens"?, "dropped"?, "in_context"?, "truncated"?, "blocked"?,
     "retries"?}
    {"kind": "end", "text", "done", "phase", "nudges", "turns"}   or   {"kind": "end", "error"}

Gemini's setup is its model, the system instruction, the tool declarations
and the generation config as the SDK serialises them for the request,
`keep_media` and `dropped_text`. `nudge` says which reminder a user message
carries (`phase`, `in_a_row`, `of`); `t` is the robot's time when it was sent.
`truncated` marks an answer cut off at `loop.MAX_OUTPUT_TOKENS`: recorded as it
came, its calls never made, and -- `in_context: false` -- never sent back;
`junk` an answer of no call and nothing but `degenerate` text, recorded and
never sent back likewise; `withheld` the positions, in its `parts`, of a
`runaway` row in an answer that made calls, which went back without them;
`retries` the service's refusals before the answer (`loop.RETRY_CODES`).
A run that ends writes `report.json` first and `end` after it, so a reader
that has seen the end finds the report; a run that raises ends on the error;
a process killed outright leaves no end at all.

The parts:

    {"type": "text", "text", "tag"?}     tag: instruction, situation, media_label, nudge
    {"type": "media", "id", "kind", "camera", "mime", "path", "note"}    path relative to the run
    {"type": "function_response", "name", "response"}
    {"type": "function_call", "name", "args"}
    {"type": "thought", "text"}
    {"type": "other", "part"}            anything else in a model's answer, as its SDK dumps it

A message of turn n is new in the n-th request; the opening is turn 1.
**Request n, rebuilt:** the user messages of turns up to n and the model
messages of turns before n, leaving out a model message marked `in_context:
false` (an answer with nothing in it, which Gemini does not keep, or one that
ran away) and the parts a model message lists in `withheld`; user
messages next to each other joined into one; and every picture or clip at a
position that a model message of turn up to n lists in `dropped` -- the media
counted 1, 2, ... in the order they were sent -- replaced by the setup's
`dropped_text`. Positions and not ids, because one picture can be sent twice:
a `wait` on an action that has already finished reports it again, pictures
and all.

A function response is the JSON the model was handed (`jsonable`): whatever is
not plain JSON became its string, as `json.dumps(default=str)` makes it -- an
`np.int64(3)` went to Gemini as "3", and is "3" here.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .loop import Turn
    from .tools import ToolResult
    from .world_model import Media

Part = dict[str, Any]

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")
#: A key of a tool's result: quoted, or snake_case before a colon.
_RESULT_KEY = re.compile(r'"\w+"\s*:|\b[a-z]+_[a-z_]+"?\s*:')


def runaway(text: str) -> bool:
    """A row of one character, the way ER2 runs away: "0" 65536 times, "_"
    3171 times (2026-10-14). Never words, whatever else the answer holds."""
    chars = Counter(c for c in text if not c.isspace())
    total = sum(chars.values())
    return total >= 20 and chars.most_common(1)[0][1] >= 0.9 * total


def degenerate(text: str) -> bool:
    """Text that is no words: a `runaway` row; a scrap of a few characters --
    "0", "0:06", "ou", "02]", "```"; a tool's result echoed back in pieces --
    'ated_m:0.012,turn_deg:{envelope:[-4,4],...}}', '"pitch_up_deg": 25.6,
    "roll_deg": 0.1, ...'; or a call written out as text instead of made --
    "api:finish{reps_done:3,success:true,...". All of them on record from
    2026-10-09 to 10-14. A short answer in Chinese ("完成了") is words.

    An answer of no call and nothing but this is not sent back (`junk`); in an
    answer that made calls only a `runaway` row is held back (`withheld`), for
    what else is no words there cannot be told from ER's own reasoning spilled
    into the text with its first letters cut -- "ata_bar in `img_008`. The bar
    is around y=535 ...", "Point 1: [440, 380] ... Left end: [440, 418]", "It
    says: to_standoff: {forward_m: -0.003, ...} ... Let's approach" -- which a
    rule counting brackets and keys took for an echo in 120 answers on record."""
    s = text.strip()
    if not s:
        return False
    if runaway(s):
        return True
    if _CJK.search(s):
        return False
    if len(s) <= 4 or re.fullmatch(r"[\d\W_]+", s) or re.match(r"api:\w+\s*\{", s):
        return True
    return "}}" in s or "]}" in s or len(_RESULT_KEY.findall(s)) >= 2


def jsonable(value: Any) -> Any:
    """What a model's API is handed: plain JSON, anything else as its string."""
    return json.loads(json.dumps(value, default=str))


def situation_part(situation: dict[str, Any]) -> Part:
    return {"type": "text", "tag": "situation",
            "text": "[situation] " + json.dumps(jsonable(situation), ensure_ascii=False)}


def media_parts(media: Media, root: Path) -> list[Part]:
    """A picture or a clip as ER is shown it: a line naming it, then the file."""
    path = Path(media.path)
    try:
        where = path.relative_to(root).as_posix()
    except ValueError:
        where = str(path)
    label = f"[{media.id}: {media.kind}{', ' + media.camera if media.camera else ''}]"
    return [{"type": "text", "tag": "media_label", "text": label},
            {"type": "media", "id": media.id, "kind": media.kind, "camera": media.camera, "mime": media.mime,
             "path": where, "note": media.note}]


def opening(instruction: str, situation: dict[str, Any], attachments: list[Media], root: Path) -> list[Part]:
    """The first message: the instruction, the situation, the first pictures."""
    parts: list[Part] = [{"type": "text", "tag": "instruction", "text": instruction}, situation_part(situation)]
    for media in attachments:
        parts += media_parts(media, root)
    return parts


def reply(results: list[ToolResult], situation: dict[str, Any], nudge: str, root: Path) -> list[Part]:
    """What answers ER's last turn: a function response for each of its calls,
    in the order made, the situation, the pictures and clips the results
    named, and the reminder. Nothing when there are neither results nor a
    reminder -- the first turn, whose request is the opening alone."""
    if not results and not nudge:
        return []
    parts: list[Part] = [{"type": "function_response", "name": r.name, "response": jsonable(r.result)}
                         for r in results]
    parts.append(situation_part(situation))
    for r in results:
        for media in r.attachments:
            parts += media_parts(media, root)
    if nudge:
        parts.append({"type": "text", "tag": "nudge", "text": nudge})
    return parts


def answer(turn: Turn) -> list[Part]:
    """ER's turn as parts: the agent's own when it has them, in the model's
    order; otherwise its words, then its calls."""
    if turn.parts is not None:
        return turn.parts
    parts: list[Part] = [{"type": "text", "text": turn.text}] if turn.text else []
    return parts + [{"type": "function_call", "name": name, "args": args} for name, args in turn.calls]


__all__ = ["Part", "answer", "degenerate", "jsonable", "media_parts", "opening", "reply", "runaway",
           "situation_part"]
