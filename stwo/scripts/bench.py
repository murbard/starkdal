#!/usr/bin/env python3
"""Benchmark the starkdal circuit: execute, prove, verify.

Measures wall time, peak RSS, and binary proof size.

Usage:
    python3 scripts/bench.py [--log-n N] [--log-blowup K] [--log-felts-per-leaf F] [--seed S]

Defaults: log_n=8, log_blowup=1, auto-picked log_felts_per_leaf, seed=0.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

import psutil

# Paths relative to project root
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_CAIRO = os.path.join(PROJ_ROOT, "src", "lib.cairo")
SCRIPTS = os.path.join(PROJ_ROOT, "scripts")

sys.path.insert(0, SCRIPTS)
from reference import deterministic_coeffs, commit, pick_log_felts_per_leaf
from proof_size import analyze as analyze_proof


def patch_constants(log_n: int, log_blowup: int, log_felts_per_leaf: int):
    """Rewrite the three const lines in src/lib.cairo."""
    with open(LIB_CAIRO) as f:
        src = f.read()
    src = re.sub(r"const LOG_N: u32 = \d+;", f"const LOG_N: u32 = {log_n};", src)
    src = re.sub(r"const LOG_BLOWUP: u32 = \d+;", f"const LOG_BLOWUP: u32 = {log_blowup};", src)
    src = re.sub(
        r"const LOG_FELTS_PER_LEAF: u32 = \d+;",
        f"const LOG_FELTS_PER_LEAF: u32 = {log_felts_per_leaf};",
        src,
    )
    with open(LIB_CAIRO, "w") as f:
        f.write(src)


def generate_args(log_n, log_blowup, log_felts_per_leaf, seed) -> str:
    """Write argument JSON file, return path."""
    coeffs = deterministic_coeffs(log_n, seed)
    result = commit(coeffs, log_n, log_blowup, log_felts_per_leaf)
    args = [len(coeffs)] + coeffs + [len(result["root_words"])] + result["root_words"]
    path = os.path.join(PROJ_ROOT, "target", "bench_args.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump([hex(x) for x in args], f)
    return path


def scarb(*args, **kwargs) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    return subprocess.run(
        ["scarb"] + list(args),
        cwd=PROJ_ROOT,
        env=env,
        capture_output=True,
        text=True,
        **kwargs,
    )


class RSSMonitor:
    """Poll a process tree for peak RSS in a background thread."""

    def __init__(self, pid: int, interval: float = 0.1):
        self.pid = pid
        self.interval = interval
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _poll(self):
        while not self._stop.is_set():
            try:
                proc = psutil.Process(self.pid)
                total = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                self.peak_rss_bytes = max(self.peak_rss_bytes, total)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            self._stop.wait(self.interval)


def timed_scarb_prove(args_path: str):
    """Run scarb prove --execute, return (wall_s, peak_rss_bytes, execution_dir)."""
    env = os.environ.copy()
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")

    t0 = time.monotonic()
    proc = subprocess.Popen(
        ["scarb", "prove", "--execute", "--arguments-file", args_path],
        cwd=PROJ_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    monitor = RSSMonitor(proc.pid)
    monitor.start()
    stdout, _ = proc.communicate()
    monitor.stop()
    wall = time.monotonic() - t0

    if proc.returncode != 0:
        print("PROVE FAILED:", stdout, file=sys.stderr)
        sys.exit(1)

    # Parse execution ID from output like "Saving output to: target/execute/starkdal/executionN"
    m = re.search(r"execution(\d+)", stdout)
    exec_id = m.group(1) if m else None

    return wall, monitor.peak_rss_bytes, exec_id, stdout


def timed_scarb_verify(exec_id: str):
    """Run scarb verify, return wall_s."""
    t0 = time.monotonic()
    r = scarb("verify", "--execution-id", exec_id)
    wall = time.monotonic() - t0
    if r.returncode != 0:
        print("VERIFY FAILED:", r.stdout, r.stderr, file=sys.stderr)
        sys.exit(1)
    return wall


def find_proof_json(exec_id: str) -> str:
    return os.path.join(PROJ_ROOT, "target", "execute", "starkdal", f"execution{exec_id}", "proof", "proof.json")


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def fmt_time(s: float) -> str:
    if s < 1:
        return f"{s * 1000:.0f} ms"
    if s < 60:
        return f"{s:.1f} s"
    return f"{s / 60:.1f} min"


def main():
    parser = argparse.ArgumentParser(description="Bench starkdal circuit")
    parser.add_argument("--log-n", type=int, default=8)
    parser.add_argument("--log-blowup", type=int, default=1)
    parser.add_argument("--log-felts-per-leaf", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log_n = args.log_n
    log_blowup = args.log_blowup
    lfpl = args.log_felts_per_leaf
    if lfpl is None:
        lfpl = pick_log_felts_per_leaf(log_n + log_blowup)
    seed = args.seed

    n_coeffs = 1 << log_n
    n_eval = 1 << (log_n + log_blowup)
    n_leaves = n_eval >> lfpl
    tree_depth = n_leaves.bit_length() - 1

    print("=" * 60)
    print("starkdal benchmark")
    print("=" * 60)
    print(f"  log_n              = {log_n}")
    print(f"  log_blowup         = {log_blowup}")
    print(f"  log_felts_per_leaf = {lfpl}")
    print(f"  seed               = {seed}")
    print(f"  n_coeffs           = {n_coeffs:,} ({n_coeffs * 32:,} B = {fmt_bytes(n_coeffs * 32)})")
    print(f"  n_eval             = {n_eval:,}")
    print(f"  n_leaves           = {n_leaves:,}")
    print(f"  tree_depth         = {tree_depth}")
    print()

    # 1. Patch constants
    print("[1/5] Patching constants ...", flush=True)
    patch_constants(log_n, log_blowup, lfpl)

    # 2. Generate arguments
    print("[2/5] Generating arguments ...", flush=True)
    args_path = generate_args(log_n, log_blowup, lfpl, seed)

    # 3. Execute (sanity check + timing)
    print("[3/5] Executing (scarb execute) ...", flush=True)
    t0 = time.monotonic()
    r = scarb("execute", "--arguments-file", args_path)
    exec_time = time.monotonic() - t0
    if r.returncode != 0:
        print("EXECUTE FAILED:", r.stdout, r.stderr, file=sys.stderr)
        sys.exit(1)

    # 4. Prove
    print("[4/5] Proving (scarb prove --execute) ...", flush=True)
    prove_wall, prove_rss, exec_id, prove_out = timed_scarb_prove(args_path)

    # 5. Verify
    print("[5/5] Verifying ...", flush=True)
    verify_wall = timed_scarb_verify(exec_id)

    # 6. Analyze proof
    proof_path = find_proof_json(exec_id)
    proof_json_size = os.path.getsize(proof_path)
    proof_analysis = analyze_proof(proof_path)
    core_keys = [k for k in proof_analysis if k.startswith("stark_proof.")]
    core_binary = sum(proof_analysis[k].total for k in core_keys)
    total_binary = proof_analysis["overall"].total

    print()
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)
    print(f"  Execute time       : {fmt_time(exec_time)}")
    print(f"  Prove time (wall)  : {fmt_time(prove_wall)}")
    print(f"  Prove peak RSS     : {fmt_bytes(prove_rss)}")
    print(f"  Verify time        : {fmt_time(verify_wall)}")
    print()
    print(f"  Proof (JSON file)  : {fmt_bytes(proof_json_size)}")
    print(f"  Proof (binary est) : {fmt_bytes(total_binary)}")
    print(f"    core stark_proof : {fmt_bytes(core_binary)}")
    print()
    print("  Core proof breakdown (binary):")
    for k in core_keys:
        c = proof_analysis[k]
        label = k.split(".", 1)[1]
        print(f"    {label:20s}  {fmt_bytes(c.total):>10s}")
    print("-" * 60)

    # Machine-readable JSON output
    out = {
        "log_n": log_n,
        "log_blowup": log_blowup,
        "log_felts_per_leaf": lfpl,
        "seed": seed,
        "n_coeffs": n_coeffs,
        "n_eval": n_eval,
        "execute_time_s": round(exec_time, 3),
        "prove_time_s": round(prove_wall, 3),
        "prove_peak_rss_bytes": prove_rss,
        "verify_time_s": round(verify_wall, 3),
        "proof_json_bytes": proof_json_size,
        "proof_binary_bytes": total_binary,
        "proof_core_binary_bytes": core_binary,
    }
    out_path = os.path.join(PROJ_ROOT, "target", "bench_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nMachine-readable results: {out_path}")


if __name__ == "__main__":
    main()
