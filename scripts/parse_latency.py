#!/usr/bin/env python3
"""Parse generation logs and emit a latency comparison table (CSV/JSON/stdout)."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DENOISE_RE = re.compile(
    r"Total denoising time:\s*([0-9]+\.?[0-9]*)\s*s", re.IGNORECASE
)
AVG_STEP_RE = re.compile(
    r"Avg step time:\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE
)
AVG_COMM_RE = re.compile(
    r"Avg comm \(all_reduce\):\s*([0-9]+\.?[0-9]*)\s*ms", re.IGNORECASE
)

STRATEGY_ORDER = ["unite", "ulysses", "ring", "fsdp", "tp"]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    denoise = DENOISE_RE.findall(text)
    avg_step = AVG_STEP_RE.findall(text)
    avg_comm = AVG_COMM_RE.findall(text)
    ok = "Finished." in text or path.with_name(path.parent.name + ".mp4").exists()
    video = path.parent / f"{path.parent.name}.mp4"
    return {
        "strategy": path.parent.name,
        "status": "success" if (denoise and video.exists()) else ("partial" if denoise else "failed"),
        "denoising_s": float(denoise[-1]) if denoise else None,
        "avg_step_ms": float(avg_step[-1]) if avg_step else None,
        "avg_comm_ms": float(avg_comm[-1]) if avg_comm else None,
        "log": str(path),
        "video": str(video) if video.exists() else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing per-strategy subfolders with generation.log",
    )
    parser.add_argument("--csv", type=str, default=None, help="Write CSV path")
    parser.add_argument("--json", type=str, default=None, help="Write JSON path")
    args = parser.parse_args()

    root = Path(args.input_dir)
    rows = []
    for sub in sorted(root.iterdir()):
        log = sub / "generation.log"
        if sub.is_dir() and log.exists():
            rows.append(parse_log(log))

    def sort_key(r):
        name = r["strategy"]
        return (STRATEGY_ORDER.index(name) if name in STRATEGY_ORDER else 99, name)

    rows.sort(key=sort_key)

    # stdout table
    headers = ["strategy", "status", "denoising_s", "avg_step_ms", "avg_comm_ms"]
    widths = [12, 10, 14, 12, 12]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        vals = [
            r["strategy"],
            r["status"],
            f"{r['denoising_s']:.3f}" if isinstance(r["denoising_s"], float) else "N/A",
            f"{r['avg_step_ms']:.1f}" if isinstance(r["avg_step_ms"], float) else "N/A",
            f"{r['avg_comm_ms']:.1f}" if isinstance(r["avg_comm_ms"], float) else "N/A",
        ]
        print("  ".join(v.ljust(w) for v, w in zip(vals, widths)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "strategy",
                    "status",
                    "denoising_s",
                    "avg_step_ms",
                    "avg_comm_ms",
                    "video",
                    "log",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"Wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
