"""Unified CLI entrypoint for BART generate-then-optimize experiments."""

import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cli.args import parse_args


def main():
    args = parse_args()
    from core.orchestration import run_experiment

    run_experiment(args)


if __name__ == "__main__":
    main()
