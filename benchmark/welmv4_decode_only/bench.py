"""Single-concurrency SGLang D-only fake-transfer benchmark, without tokenizer."""
import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request


def run(
    url, seed, input_len, output_len, *, dp_rank=0, bootstrap_room=None, rid=None,
    ready=None, timeout=180,
):
    ids = random.Random(seed).choices(range(100, 20000), k=input_len)
    payload = {
        "input_ids": ids, "stream": True, "bootstrap_host": "2.2.2.2",
        "bootstrap_room": seed if bootstrap_room is None else bootstrap_room,
        "routed_dp_rank": dp_rank,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": output_len, "ignore_eos": True,
        },
    }
    if rid is not None:
        payload["rid"] = rid
    request = urllib.request.Request(url + "/generate", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    if ready is not None:
        ready.wait(timeout)
    start = time.monotonic_ns()
    first = last = seen = prompt = None
    token_ids = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            t = time.monotonic_ns()
            if not line.startswith(b"data:"):
                continue
            value = line[5:].strip()
            if value == b"[DONE]":
                break
            event = json.loads(value)
            if "error" in event:
                raise RuntimeError(event["error"])
            info = event["meta_info"]
            count = info["completion_tokens"]
            prompt = info.get("prompt_tokens")
            if count > (seen or 0):
                ids = event.get("output_ids", [])
                if len(ids) == count:
                    token_ids = ids
                elif len(ids) == count - (seen or 0):
                    token_ids.extend(ids)
                else:
                    raise ValueError(f"Missing output IDs: {len(ids)=} {count=} {seen=}")
                if first is None:
                    if count != 1:
                        raise ValueError(f"First stream chunk had {count} tokens")
                    first = t
                last, seen = t, count
    if seen != output_len or prompt != input_len or len(token_ids) != output_len:
        raise ValueError(f"Expected {input_len}/{output_len}, saw {prompt}/{seen}/{len(token_ids)} IDs")
    elapsed = (last - first) / 1e9
    return {"seed": seed, "input_tokens": prompt, "output_tokens": seen,
            "fake_ttft_ms": (first - start) / 1e6,
            "tpot_ms": 1000 * elapsed / (seen - 1),
            "decode_tokens_per_s": (seen - 1) / elapsed,
            "output_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(), "output_token_ids": token_ids}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:16699")
    parser.add_argument("--input", type=int, default=16500)
    parser.add_argument("--output", type=int, default=150)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2000)
    args = parser.parse_args()
    if args.input < 1 or args.output < 2 or args.rounds < 1:
        parser.error("Require input >= 1, output >= 2, rounds >= 1")
    rows = []
    for index in range(args.rounds + 1):
        row = run(args.url, args.seed + index, args.input, args.output)
        print(json.dumps({"warmup": index == 0, **row}), flush=True)
        if index:
            rows.append(row)
    print(json.dumps({"rounds": len(rows), "concurrency": 1, **{
        f"mean_{key}": statistics.mean(x[key] for x in rows)
        for key in ("fake_ttft_ms", "tpot_ms", "decode_tokens_per_s")
    }}), flush=True)
