"""CLI for Task023 identity, construction, validation and post-hoc analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import task023_average_rescue_causal_ablation as task023


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("identity", "construct-original", "construct", "validate", "reuse-original-validation", "analyze", "integrity"), required=True)
    parser.add_argument("--variant", choices=task023.VARIANTS, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--parameters-before", type=int)
    for name in ("task014_root", "task016_root", "task017_root", "task018_root", "task019_root", "task020_root", "task021_root", "task022_root"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    args = parser.parse_args(argv)
    if args.mode == "identity":
        required = (args.task014_root, args.task016_root, args.task017_root, args.task018_root,
                    args.task019_root, args.task020_root, args.task021_root, args.task022_root,
                    args.checkpoint, args.repo_root)
        if any(value is None for value in required):
            parser.error("identity requires all task roots, --checkpoint and --repo-root")
        task023.verify_identity(task014_root=args.task014_root, task016_root=args.task016_root,
                                task017_root=args.task017_root, task018_root=args.task018_root,
                                task019_root=args.task019_root, task020_root=args.task020_root,
                                task021_root=args.task021_root, task022_root=args.task022_root,
                                output_dir=args.output_dir, checkpoint=args.checkpoint,
                                repo_root=args.repo_root)
    elif args.mode == "construct-original":
        if args.task016_root is None or args.task017_root is None or args.task018_root is None or args.parameters_before is None:
            parser.error("construct-original requires Task016/17/18 roots and --parameters-before")
        task023.construct_original(task016_root=args.task016_root, task017_root=args.task017_root,
                                   task018_root=args.task018_root, output_dir=args.output_dir,
                                   parameters_before=args.parameters_before)
    elif args.mode == "construct":
        if args.variant not in task023.RESCUE_VARIANTS or any(value is None for value in (args.task014_root, args.task016_root, args.task017_root, args.task018_root)):
            parser.error("construct requires a rescue variant and Task014/16/17/18 roots")
        task023.construct_variant(variant=args.variant, task014_root=args.task014_root,
                                  task016_root=args.task016_root, task017_root=args.task017_root,
                                  task018_root=args.task018_root, output_dir=args.output_dir,
                                  device=args.device)
    elif args.mode == "validate":
        if args.variant not in task023.RESCUE_VARIANTS:
            parser.error("validate requires V1/V2/V3")
        task023.validate_variant(args.output_dir, args.variant, args.device)
    elif args.mode == "reuse-original-validation":
        if args.task019_root is None:
            parser.error("reuse-original-validation requires --task019-root")
        task023.reuse_original_validation(args.output_dir, args.task019_root)
    elif args.mode == "analyze":
        if any(value is None for value in (args.task019_root, args.task020_root, args.task021_root, args.task022_root)):
            parser.error("analyze requires Task019/20/21/22 roots")
        task023.posthoc_analysis(output_dir=args.output_dir, task019_root=args.task019_root,
                                 task020_root=args.task020_root, task021_root=args.task021_root,
                                 task022_root=args.task022_root)
    else:
        required = (args.task014_root, args.task016_root, args.task017_root, args.task018_root,
                    args.task019_root, args.task020_root, args.task021_root, args.task022_root,
                    args.checkpoint, args.repo_root)
        if any(value is None for value in required):
            parser.error("integrity requires all task roots, --checkpoint and --repo-root")
        task023.final_integrity_run(task014_root=args.task014_root, task016_root=args.task016_root,
                                    task017_root=args.task017_root, task018_root=args.task018_root,
                                    task019_root=args.task019_root, task020_root=args.task020_root,
                                    task021_root=args.task021_root, task022_root=args.task022_root,
                                    output_dir=args.output_dir, checkpoint=args.checkpoint,
                                    repo_root=args.repo_root, device=args.device)


if __name__ == "__main__":
    main()
