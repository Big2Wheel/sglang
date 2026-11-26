"""Verify DP8EP8 fake Decode concurrency against an isolated localhost service."""

import argparse
import ast
import json
import random
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from bench import run

DP = 8
BUCKETS = (1, 2, 4, 8)
CASES = {
    str(n): tuple(i % DP for i in range(n)) for n in (1, 4, 8, 16, 32, 64)
}
CASES["skew"] = (0, 0, 0, 2, 2, 3, 7, 7)  # Max 3 rows, padded B4.
CASES["skew8"] = (0, 0, 0, 0, 0, 2, 7, 7)  # Max 5 rows, padded B8.
RANK = re.compile(r"\bDP(\d+) TP(\d+) EP(\d+)\]")
STEP = re.compile(
    r"Decode graph replay: worker=target key_size=(\d+) \(bs\) "
    r"mode=(DECODE|IDLE) raw_bs=(\d+)\s*$"
)
SHAPE = r"ShapeKey\(size=(\d+), stream_idx=None, variant_label=None\)"
COVERAGE = re.compile(
    rf"welm-mk-target recorded shape={SHAPE} layer_launches\(total,fused\)="
    r"(.*?) coverage=(.*)$"
)
REPLAY = re.compile(rf"welm-mk-target replay shape={SHAPE} count=(\d+) backend=(\S+)\s*$")


def rank_of(line):
    matches = RANK.findall(line)
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous DP/TP/EP rank: {line[:180]}")
    dp, tp, ep = map(int, matches[0])
    if dp != tp or dp != ep or not 0 <= dp < DP:
        raise ValueError(f"Expected matching DP/TP/EP ranks in 0..7: {line[:180]}")
    return dp


def audit_steps(lines, routes, min_steps):
    counts = Counter(routes)
    expected = tuple(counts[rank] for rank in range(DP))
    expected_bucket = next(bucket for bucket in BUCKETS if bucket >= max(expected))
    records = [[] for _ in range(DP)]
    for line in lines:
        if "Decode graph replay:" not in line:
            continue
        match = STEP.search(line)
        if match is None:
            raise ValueError(f"Unexpected graph step: {line[:180]}")
        rank = rank_of(line)
        bucket, mode, rows = int(match[1]), match[2], int(match[3])
        if bucket not in BUCKETS or not 0 <= rows <= min(bucket, expected[rank]):
            raise ValueError(f"Invalid bucket or unexpected rows at rank {rank}: {line[:180]}")
        if (rows == 0) != (mode == "IDLE"):
            raise ValueError(f"Inconsistent Decode/IDLE row marker: {line[:180]}")
        records[rank].append((bucket, rows))
    lengths = [len(row) for row in records]
    if not lengths[0] or len(set(lengths)) != 1:
        raise ValueError(f"Missing or unbalanced rank telemetry: {lengths}")
    streak = best = matching = peak = 0
    buckets = Counter()
    for step in zip(*records, strict=True):
        bucket = step[0][0]
        if any(key != bucket for key, _ in step):
            raise ValueError(f"Collective ranks used inconsistent buckets: {step}")
        rows = tuple(count for _, count in step)
        buckets[bucket] += 1
        peak = max(peak, sum(rows))
        if rows == expected and bucket == expected_bucket:
            streak += 1
            matching += 1
            best = max(best, streak)
        else:
            streak = 0
    return {"qualified": best >= min_steps, "expected_rows": expected,
            "expected_bucket": expected_bucket, "observed_peak_rows": peak,
            "matching_steps": matching, "longest_matching_streak": best,
            "steps_by_bucket": dict(sorted(buckets.items()))}


def audit_backend(lines, backend, used_buckets):
    recorded, replayed = set(), set()
    for line in lines:
        if "welm-mk-target eager fallback" in line or "WeLM fixed ACL graph captured" in line:
            raise ValueError(f"Unexpected fallback/wrapper around direct replay: {line[:180]}")
        if "welm-mk-target recorded " in line:
            match = COVERAGE.search(line)
            if backend != "megakernel-native" or match is None:
                raise ValueError(f"Unexpected or unreadable MK capture: {line[:180]}")
            key = rank_of(line), int(match[1])
            layers, summary = ast.literal_eval(match[2]), ast.literal_eval(match[3])
            if (key in recorded or not isinstance(layers, dict) or set(layers) != set(range(48))
                    or not all(total >= fused > 0 for total, fused in layers.values())
                    or not isinstance(summary, dict) or summary.get("torch_boundaries") != []):
                raise ValueError(f"Incomplete or duplicate native capture at {key}")
            recorded.add(key)
        if "welm-mk-target replay " in line:
            match = REPLAY.search(line)
            if backend != "megakernel-native" or match is None or match[3] != "mk-direct":
                raise ValueError(f"Unexpected graph executor: {line[:180]}")
            key = rank_of(line), int(match[1])
            if key not in recorded or int(match[2]) < 1:
                raise ValueError(f"Replay without a valid native capture at {key}")
            replayed.add(key)
    wanted = {(rank, bucket) for rank in range(DP) for bucket in used_buckets}
    if backend == "megakernel-native" and wanted - replayed:
        raise ValueError(f"Unproven MK direct replays: {sorted(wanted - replayed)}")


def log_marker(path):
    record = path.stat()
    return record.st_dev, record.st_ino, record.st_size


def read_slice(path, marker):
    end = log_marker(path)
    if end[:2] != marker[:2] or end[2] < marker[2]:
        raise ValueError("Server log rotated or truncated during the request wave")
    with path.open("rb") as stream:
        stream.seek(marker[2])
        return stream.read(end[2] - marker[2]).decode(errors="replace").splitlines()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:16698")
    parser.add_argument("--serve-log", type=Path, required=True)
    parser.add_argument("--backend", choices=("megakernel-native", "native"), default="megakernel-native")
    allowed = ",".join(CASES)
    parser.add_argument("--cases", default=allowed, help=f"Ordered subset: {allowed}")
    parser.add_argument("--input", type=int, default=1024)
    parser.add_argument("--output", type=int, default=32)
    parser.add_argument("--waves", type=int, default=3, help="Max attempts per case to realize full rows")
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    cases = args.cases.split(",")
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        parser.error(f"Use distinct cases from {allowed}")
    if min(args.input, args.waves) < 1 or not 2 <= args.min_steps < args.output or not args.timeout > 0:
        parser.error("Require positive input/waves/timeout and 2 <= min-steps < output")
    room_base = random.SystemRandom().randrange(1 << 48)
    successful, used_buckets = [], set()
    for case in cases:
        routes = CASES[case]
        for wave in range(args.waves):
            # Each wave gets unique fake handoff room and RID, including retries.
            start = log_marker(args.serve_log)
            ready = Barrier(len(routes))
            with ThreadPoolExecutor(max_workers=len(routes)) as pool:
                pending = []
                for slot, rank in enumerate(routes):
                    room_base += 1
                    pending.append(pool.submit(run, args.url.rstrip("/"), 2000 + slot,
                                               args.input, args.output, dp_rank=rank,
                                               bootstrap_room=room_base,
                                               rid=f"welm-dp8-{room_base}", ready=ready,
                                               timeout=args.timeout))
                rows = [future.result() for future in pending]
            time.sleep(0.25)  # Let the owning workers flush their final log lines.
            evidence = audit_steps(read_slice(args.serve_log, start), routes, args.min_steps)
            print(json.dumps({"case": case, "wave": wave, "offered": len(routes),
                              "completed": len(rows), **evidence,
                              "mean_fake_ttft_ms": statistics.mean(r["fake_ttft_ms"] for r in rows),
                              "mean_tpot_ms": statistics.mean(r["tpot_ms"] for r in rows)}), flush=True)
            if evidence["qualified"]:
                successful.append(case)
                used_buckets.update(evidence["steps_by_bucket"])
                break
    with args.serve_log.open(encoding="utf-8", errors="replace") as stream:
        audit_backend(stream, args.backend, used_buckets)
    print(json.dumps({"backend": args.backend, "passed": successful,
                      "replay_buckets": sorted(used_buckets), "requested": cases,
                      "alignment": "per-rank ordinal within each isolated wave"}))
    return 0 if successful == cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
