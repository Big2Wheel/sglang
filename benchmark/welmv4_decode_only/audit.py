"""Verify the timed decode really exercised the full fixed MegaKernel path."""

import argparse
import ast
import json
import re
from pathlib import Path

RANK = re.compile(r"DP(\d+) TP\d+ EP\d+\]")
ACL_TOPOLOGY = re.compile(r"WeLM fixed ACL graph captured: scopes=(\d+) steps=(\d+)")
DIRECT_TOPOLOGY = re.compile(
    r"WeLM MK direct replay: scopes=(\d+) steps=(\d+) acl_graph=False"
)


def audit(log: Path, *, direct: bool = False) -> dict:
    coverage, replays = set(), set()
    topology_by_rank = {}
    expected_backend = "mk-direct" if direct else "acl"
    expected_topology = DIRECT_TOPOLOGY if direct else ACL_TOPOLOGY
    forbidden_topology = ACL_TOPOLOGY if direct else DIRECT_TOPOLOGY
    for line in log.read_text(errors="replace").splitlines():
        if "welm-mk-target eager fallback" in line:
            raise RuntimeError(line)
        if not (match := RANK.search(line)):
            continue
        rank = int(match[1])
        if forbidden_topology.search(line):
            unexpected = "ACL Graph" if direct else "MK direct replay"
            raise RuntimeError(f"Unexpected {unexpected} on rank {rank}: {line}")
        if topology := expected_topology.search(line):
            scopes, steps = map(int, topology.groups())
            if rank in topology_by_rank or not 0 < scopes <= steps:
                raise RuntimeError(
                    f"Invalid {expected_backend} topology on rank {rank}: {line}"
                )
            topology_by_rank[rank] = {"scopes": scopes, "steps": steps}
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
            if (
                "backend=" in line or direct
            ) and f"backend={expected_backend}" not in line:
                raise RuntimeError(f"Unexpected replay backend on rank {rank}: {line}")
            replays.add(rank)
    expected = set(range(4))
    if coverage != expected or set(topology_by_rank) != expected or replays != expected:
        mode = "MK direct" if direct else "ACL"
        raise RuntimeError(
            f"Incomplete {mode} run: {coverage=}, {topology_by_rank=}, {replays=}"
        )
    return {
        "verified_ranks": sorted(expected),
        "layers_per_rank": 48,
        "backend": expected_backend,
        "topology_by_rank": topology_by_rank,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.log, direct=args.direct)))
