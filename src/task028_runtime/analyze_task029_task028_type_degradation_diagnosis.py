"""Offline analyzer entry point for Task029 generated replay artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import task029_task028_type_degradation_diagnosis as task029


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task028-root", type=Path, required=True)
    parser.add_argument("--replay", type=Path, default=None,
                        help="Task029 replay.json (the default is OUTPUT/replay.json)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("task029_task028_type_degradation_diagnosis"))
    args = parser.parse_args(argv)
    replay_path = args.replay or (args.output_dir / "replay_data.json")
    replay = task029.read_json(replay_path)
    # replay.json intentionally omits large arrays.  This command is intended
    # for the in-process run pipeline; retaining the explicit error prevents
    # silently producing incomplete tables from a compact artifact.
    required = ("snapshot_rows", "selected_rows", "all_units")
    missing = [key for key in required if key not in replay]
    if missing:
        raise RuntimeError("replay.json is compact; rerun Task029 --mode run to analyze: " + str(missing))
    result = task029.analyze_replay(task028_root=args.task028_root,
                                    replay=replay, output_dir=args.output_dir)
    task029.make_figures(args.output_dir, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
