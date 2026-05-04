#!/usr/bin/env python3
"""Analyze a Stwo proof JSON and estimate its binary size.

The JSON proof encodes everything as nested lists/dicts of integers.  A faithful
binary encoding stores:
  - each integer in its natural width (1/2/4/8 bytes)
  - each array with a 4-byte u32 length prefix
  - dict keys are implicit (schema-defined), so zero cost

This gives a realistic lower bound on wire size; an actual binary format might
add version/tag bytes but those are negligible.
"""
import json
import sys
from collections import defaultdict


def int_width(v: int) -> int:
    """Minimum bytes to encode a non-negative integer."""
    if v < 0:
        v = -v
    if v < (1 << 8):
        return 1
    if v < (1 << 16):
        return 2
    if v < (1 << 32):
        return 4
    if v < (1 << 64):
        return 8
    if v < (1 << 128):
        return 16
    return 32


class SizeCounter:
    def __init__(self):
        self.int_bytes = 0
        self.int_count = 0
        self.array_headers = 0  # 4 bytes each
        self.string_bytes = 0
        self.by_width = defaultdict(int)  # width -> count

    def walk(self, obj):
        if isinstance(obj, int):
            w = int_width(obj)
            self.int_bytes += w
            self.int_count += 1
            self.by_width[w] += 1
        elif isinstance(obj, float):
            self.int_bytes += 8
            self.int_count += 1
            self.by_width[8] += 1
        elif isinstance(obj, str):
            self.string_bytes += len(obj)
        elif isinstance(obj, list):
            self.array_headers += 4
            for item in obj:
                self.walk(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                self.walk(v)
        # null / bool: 0 bytes

    @property
    def total(self) -> int:
        return self.int_bytes + self.array_headers + self.string_bytes

    def report(self, label: str = "") -> str:
        lines = []
        if label:
            lines.append(f"  {label}:")
        lines.append(f"    integers: {self.int_count:,} ({self.int_bytes:,} B)")
        for w in sorted(self.by_width):
            lines.append(f"      {w}-byte: {self.by_width[w]:,}")
        lines.append(f"    array headers: {self.array_headers:,} ({self.array_headers:,} B)")
        if self.string_bytes:
            lines.append(f"    strings: {self.string_bytes:,} B")
        lines.append(f"    total: {self.total:,} B ({self.total / 1024:.1f} KB)")
        return "\n".join(lines)


def analyze(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    results = {}

    # Overall
    overall = SizeCounter()
    overall.walk(data)
    results["overall"] = overall

    # Per top-level key
    if isinstance(data, dict):
        for k, v in data.items():
            c = SizeCounter()
            c.walk(v)
            results[k] = c

    # stark_proof breakdown
    sp = data.get("stark_proof", {})
    if isinstance(sp, dict):
        for k, v in sp.items():
            c = SizeCounter()
            c.walk(v)
            results[f"stark_proof.{k}"] = c

    return results


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "target/execute/starkdal/execution15/proof/proof.json"
    results = analyze(path)

    overall = results.pop("overall")
    print(f"Binary proof size estimate: {overall.total:,} B ({overall.total / 1024:.1f} KB)")
    print()

    # Print stark_proof breakdown first (the core proof)
    sp_keys = [k for k in results if k.startswith("stark_proof.")]
    sp_total = sum(results[k].total for k in sp_keys)
    print(f"stark_proof (core proof): {sp_total:,} B ({sp_total / 1024:.1f} KB)")
    for k in sp_keys:
        c = results[k]
        print(f"  {k.split('.', 1)[1]:20s}  {c.total:>8,} B  ({c.total / 1024:>7.1f} KB)")
    print()

    # Other top-level keys
    other_keys = [k for k in results if not k.startswith("stark_proof.") and k != "stark_proof"]
    if other_keys:
        print("Other sections:")
        for k in other_keys:
            c = results[k]
            print(f"  {k:30s}  {c.total:>8,} B  ({c.total / 1024:>7.1f} KB)")

    return overall.total


if __name__ == "__main__":
    main()
