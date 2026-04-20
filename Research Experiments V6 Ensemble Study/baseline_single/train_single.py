"""
V6 compute-matched single-member baseline trainer.

Literature-required control (Lin et al. 2024 "Curse of Diversity"): trains a
single HierSmartPlayer for 5M battles — 1/6 of the K=30 × 1M ensemble compute
budget. Answers the reviewer question "would we just be better off
concentrating the compute in one agent?"

Uses the same V6 HierSmartPlayer and hp_001 config as each ensemble member;
only the battle budget differs.
"""

import os
import sys
import asyncio

# Ensure V6's root is on sys.path so `from shared.X import ...` resolves to
# V6's own shared/ package.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V6_DIR = os.path.dirname(_THIS_DIR)
if _V6_DIR not in sys.path:
    sys.path.insert(0, _V6_DIR)

from shared.train_common import run_training, parse_train_args   # noqa: E402
from model_ensemble.hier_smart_player import HierSmartPlayer     # noqa: E402


if __name__ == "__main__":
    args = parse_train_args()
    model_dir = args.output_dir or _THIS_DIR
    asyncio.run(run_training(HierSmartPlayer, args, model_dir))
