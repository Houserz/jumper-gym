#!/usr/bin/env python3
"""Cut both claws' links into the convex pieces `claws.py` loads.

    python tasks/jumper/pull_up/tools/cut_claws.py            # writes meshes/
    python tasks/jumper/pull_up/tools/cut_claws.py --check    # compares with what is there

The pieces come from the **visual** meshes: the `_col` meshes in the asset are
already hulls, with no pocket left in them to cut. The method is
`tasks/jumper/five_foot/tools/jaw_decompose.py`'s -- slabs along the link's first
principal axis, overlapping by 0.15 of their width, each hull decimated to 48
support directions -- with the counts in `claws.CUTS`. Re-run it after the claw's
mesh changes; `--check` says whether the committed pieces still match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh

from tasks.jumper.common.assets import JUMPER_XML
from tasks.jumper.five_foot.tools.jaw_decompose import decompose
from tasks.jumper.pull_up.claws import CLAW_PIECES, MESH_DIR

#: The visual meshes, beside the URDF the asset was built from.
VISUAL = JUMPER_XML.parent / "urdf" / "jumper" / "meshes" / "visual"

OVERLAP = 0.15
KEEP = 48


def cut(link: str, count: int) -> list[trimesh.Trimesh]:
    mesh = trimesh.load(VISUAL / f"{link}.stl", force="mesh")
    return decompose(mesh, count, OVERLAP, KEEP)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="compare freshly cut pieces with meshes/ instead of writing")
    args = parser.parse_args()
    MESH_DIR.mkdir(exist_ok=True)
    stale = []
    for link, count in CLAW_PIECES.items():
        for i, piece in enumerate(cut(link, count)):
            path: Path = MESH_DIR / f"{link}_col{i}.STL"
            if args.check:
                have = trimesh.load(path, force="mesh") if path.is_file() else None
                same = (have is not None and len(have.vertices) == len(piece.vertices)
                        and np.allclose(np.sort(have.vertices, axis=0),
                                        np.sort(piece.vertices, axis=0), atol=1e-6))
                if not same:
                    stale.append(path.name)
            else:
                piece.export(path)
    if args.check:
        if stale:
            print(f"{len(stale)} piece(s) differ from a fresh cut: {', '.join(stale)}")
            return 1
        print(f"every piece under {MESH_DIR} matches a fresh cut")
        return 0
    print(f"wrote {sum(CLAW_PIECES.values())} pieces under {MESH_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
