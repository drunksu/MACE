#!/usr/bin/env python3
"""
Preprocessing script.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
from mace_rl.data.preprocess import preprocess
from mace_rl.utils.config import load_config
from mace_rl.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/preprocess.yaml',
                        help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level')
    args = parser.parse_args()

    setup_logging(level=args.log_level)
    config = load_config(args.config)
    preprocess(config)


if __name__ == '__main__':
    main()