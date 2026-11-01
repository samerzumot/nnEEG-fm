"""Command Line Interface for nnEEG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nneeg.contracts import get_contract, list_contracts
from nneeg.fingerprint import fingerprint_dataset
from nneeg.planner import generate_plan


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="nneeg",
        description="nnEEG: Automated EEG dataset fingerprinting and foundation model adaptation planner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: fingerprint
    fp_parser = subparsers.add_parser(
        "fingerprint",
        help="Extract dataset fingerprint (sampling rates, montage, line noise, amplitude stats).",
    )
    fp_parser.add_argument(
        "path",
        type=str,
        help="Path to an EEG file (.edf, .bdf, .vhdr, .set, .fif) or directory of EEG files.",
    )
    fp_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional path to save fingerprint JSON artifact.",
    )

    # Subcommand: plan
    plan_parser = subparsers.add_parser(
        "plan",
        help="Generate a deterministic PipelinePlan matching a dataset to an FM contract.",
    )
    plan_parser.add_argument(
        "path",
        type=str,
        help="Path to an EEG file (.edf, .bdf, .vhdr, .set, .fif) or directory of EEG files.",
    )
    plan_parser.add_argument(
        "--fm",
        type=str,
        required=True,
        help=f"Target foundation model contract name. Available: {list_contracts()}",
    )
    plan_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional path to save plan JSON artifact.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "fingerprint":
            print(f"Fingerprinting dataset at: {args.path} ...", file=sys.stderr)
            fp = fingerprint_dataset(args.path)
            json_output = fp.to_json(indent=2)
            if args.output:
                out_path = Path(args.output).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json_output, encoding="utf-8")
                print(f"Fingerprint saved to: {out_path}", file=sys.stderr)
            else:
                print(json_output)
            return 0

        elif args.command == "plan":
            print(f"Fingerprinting dataset at: {args.path} ...", file=sys.stderr)
            fp = fingerprint_dataset(args.path)
            contract = get_contract(args.fm)
            print(f"Generating plan for FM contract '{contract.name}' ...", file=sys.stderr)
            plan = generate_plan(fp, contract)
            json_output = plan.to_json(indent=2)
            if args.output:
                out_path = Path(args.output).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json_output, encoding="utf-8")
                print(f"Pipeline plan saved to: {out_path}", file=sys.stderr)
            else:
                print(json_output)
            return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
