"""Run WidePresent's external TicToc B-vs-C gate against the official repository.

B = official TicToc timestamp condition
C = same model / same samples, but timestamps are hidden from the model and replaced by
    label-blind deterministic relative-time state from temporal_kernel.py.

This wrapper intentionally delegates inference and scoring to the OFFICIAL TicToc
scripts.  It does not reimplement their model handlers or metric.

Requirements
------------
1. Clone https://github.com/chengez/TicToc
2. Install that repository's requirements.
3. For API models, configure the API environment variables required by TicToc.
4. Run this script from a WidePresent checkout.

Example
-------
python experiments/run_tictoc_b_vs_c.py \
    --tictoc-dir ../TicToc \
    --model gpt-4.1-mini-2025-04-14-FC \
    --data merged_fully_labeled_data.json

The script prints the exact commands before running them and reports the official
Normalized Alignment Rate for B and C plus C-B.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

# Reuse the already-reviewed adapter rather than duplicating temporal rendering logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tictoc_temporal_kernel_adapter import prepare


NORMALIZED_RE = re.compile(r"Normalized Alignment Rate:\s*([0-9.]+)%")


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    print("\n$ " + " ".join(command))
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    print(completed.stdout)
    return completed.stdout


def parse_alignment(text: str) -> float:
    match = NORMALIZED_RE.search(text)
    if match is None:
        raise RuntimeError("could not parse official Normalized Alignment Rate")
    return float(match.group(1)) / 100.0


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tictoc-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini-2025-04-14-FC",
        help="exact model key from TicToc/inference/model_map.py",
    )
    parser.add_argument(
        "--data",
        default="merged_fully_labeled_data.json",
        help="official TicToc JSON file, relative to --tictoc-dir unless absolute",
    )
    parser.add_argument(
        "--output-dir",
        default="widepresent_tictoc_outputs",
        help="output directory inside TicToc unless absolute",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for the official TicToc scripts",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="create/validate condition C without making model API calls",
    )
    args = parser.parse_args()

    tictoc = args.tictoc_dir.resolve()
    required = [
        tictoc / "eval_from_api.py",
        tictoc / "get_metric.py",
        tictoc / "inference" / "model_map.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("not an official TicToc checkout; missing: " + ", ".join(missing))

    raw_data = Path(args.data)
    if not raw_data.is_absolute():
        raw_data = tictoc / raw_data
    raw_data = raw_data.resolve()
    if not raw_data.exists():
        raise FileNotFoundError(raw_data)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = tictoc / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition C lives beside outputs so the official scripts can consume it by path.
    derived_data = output_dir / f"{raw_data.stem}__widepresent_kernel_v1.json"

    original = load_json(raw_data)
    if isinstance(original, dict):
        original = [original]
    derived = prepare(original, expand_variants=False)

    if len(derived) != len(original):
        raise AssertionError("condition C changed sample count")
    if [item.get("id") for item in derived] != [item.get("id") for item in original]:
        raise AssertionError("condition C changed sample IDs/order")

    with derived_data.open("w", encoding="utf-8") as handle:
        json.dump(derived, handle, indent=2, ensure_ascii=False)

    print(f"prepared condition C: {derived_data}")
    print(f"samples: {len(derived)}")
    print("B exposes official absolute timestamps; C hides them and exposes only deterministic clock-derived state.")

    if args.prepare_only:
        return

    env = os.environ.copy()

    # The official repository currently has separate API/local runners.  This wrapper
    # uses eval_from_api.py because the external gate was designed around an API model.
    b_infer = [
        args.python,
        "eval_from_api.py",
        "--model",
        args.model,
        "--data",
        str(raw_data),
        "--use_time_stamp",
        "--output_dir",
        str(output_dir),
    ]
    c_infer = [
        args.python,
        "eval_from_api.py",
        "--model",
        args.model,
        "--data",
        str(derived_data),
        "--output_dir",
        str(output_dir),
    ]

    run(b_infer, cwd=tictoc, env=env)
    run(c_infer, cwd=tictoc, env=env)

    b_metric = [
        args.python,
        "get_metric.py",
        "--model",
        args.model,
        "--data",
        str(raw_data),
        "--use_time_stamp",
        "--output_dir",
        str(output_dir),
    ]
    c_metric = [
        args.python,
        "get_metric.py",
        "--model",
        args.model,
        "--data",
        str(derived_data),
        "--output_dir",
        str(output_dir),
    ]

    b_text = run(b_metric, cwd=tictoc, env=env)
    c_text = run(c_metric, cwd=tictoc, env=env)

    b = parse_alignment(b_text)
    c = parse_alignment(c_text)

    print("\nTICTOC B-vs-C")
    print(f"B official timestamps : {b:.4%}")
    print(f"C derived kernel      : {c:.4%}")
    print(f"C - B                 : {(c - b):+.4%}")
    print()
    if c > b:
        print("Positive external gate: derived label-blind temporal state beat passive timestamp text.")
    else:
        print("Negative external gate: passive timestamp text matched or beat the derived temporal state.")


if __name__ == "__main__":
    main()
