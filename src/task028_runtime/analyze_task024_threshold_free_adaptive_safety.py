"""Command-line entry point for Task024 identity, construction and analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import task024_threshold_free_adaptive_safety as task024


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "construct", "validate", "analyze"), required=True)
    parser.add_argument("--variant", choices=task024.VARIANTS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    for name in ("task023_root", "task014_root", "task016_root", "task017_root",
                 "task018_root", "task019_root", "task020_root", "task021_root",
                 "task022_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "identity":
        values = (args.task023_root, args.task014_root, args.task016_root,
                  args.task017_root, args.task018_root, args.task019_root,
                  args.task020_root, args.task021_root, args.task022_root,
                  args.checkpoint, args.repo_root)
        if any(value is None for value in values):
            parser.error("identity requires all task roots, checkpoint and repo root")
        task024.verify_identity(
            task023_root=args.task023_root, task014_root=args.task014_root,
            task016_root=args.task016_root, task017_root=args.task017_root,
            task018_root=args.task018_root, task019_root=args.task019_root,
            task020_root=args.task020_root, task021_root=args.task021_root,
            task022_root=args.task022_root, output_dir=args.output_dir,
            checkpoint=args.checkpoint, repo_root=args.repo_root,
        )
    elif args.mode == "construct":
        values = (args.variant, args.task023_root, args.task014_root,
                  args.task016_root, args.task017_root, args.task018_root)
        if any(value is None for value in values):
            parser.error("construct requires variant and Task014/16/17/18/23 roots")
        task024.construct_variant(
            variant=args.variant, task023_root=args.task023_root,
            task014_root=args.task014_root, task016_root=args.task016_root,
            task017_root=args.task017_root, task018_root=args.task018_root,
            output_dir=args.output_dir, device=args.device,
        )
    elif args.mode == "validate":
        if args.variant is None:
            parser.error("validate requires variant")
        task024.validate_variant(args.output_dir, args.variant, args.device)
    else:
        if args.task023_root is None or args.task020_root is None:
            parser.error("analyze requires Task023 and Task020 roots")
        task024.analyze(output_dir=args.output_dir, task023_root=args.task023_root,
                        task020_root=args.task020_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
