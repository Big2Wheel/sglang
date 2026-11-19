"""Verify the timed decode really exercised the full fixed MegaKernel path."""

import argparse
import ast
import json
import re
from pathlib import Path

RANK = re.compile(r"DP(\d+) TP\d+ EP\d+\]")


def audit(log: Path) -> dict:
    coverage, wrappers, replays = set(), set(), set()
    for line in log.read_text(errors="replace").splitlines():
        if "welm-mk-target eager fallback" in line:
            raise RuntimeError(line)
        if not (match := RANK.search(line)):
            continue
        rank = int(match[1])
        if "WeLM fixed ACL graph captured: scopes=147 steps=1012" in line:
            if rank in wrappers:
                raise RuntimeError(f"Duplicate ACL graph on rank {rank}")
            wrappers.add(rank)
        if "ShapeKey(size=1," not in line:
            continue
        if "welm-mk-target recorded " in line:
            counts, summary = line.split("layer_launches(total,fused)=", 1)[1].split(
                " coverage=", 1
            )
            counts, summary = ast.literal_eval(counts), ast.literal_eval(summary)
            if (
                list(counts) != list(range(48))
                or not all(total >= fused > 0 for total, fused in counts.values())
                or summary["torch_boundaries"]
            ):
                raise RuntimeError(f"Incomplete device coverage on rank {rank}")
            coverage.add(rank)
        if "welm-mk-target replay " in line:
            replays.add(rank)
    expected = set(range(4))
    if not all(ranks == expected for ranks in (coverage, wrappers, replays)):
        raise RuntimeError(f"Incomplete four-rank run: {coverage=}, {wrappers=}, {replays=}")
    return {"verified_ranks": sorted(expected), "layers_per_rank": 48, "scopes": 147, "steps": 1012}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.log)))
