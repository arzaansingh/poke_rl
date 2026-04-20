"""
V6 ensemble member trainer.

Trains one ensemble member using V6's `HierSmartPlayer` at the best V5 config
(hp_001: alpha=0.1, gamma=0.99, lambda=0.7, fixed epsilon=0.05).

This file is a thin wrapper — all real logic lives in V6's
`shared.train_common.run_training`. The orchestrator `run_ensemble.py` spawns
one subprocess per member pointing at this script, passing CLI args for
--seed, --batch_size (= BATTLES_PER_MEMBER), --port, --output_dir, and --pool.
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
    # `run_training` writes logs/, models/, and a live_status JSON into
    # `args.output_dir` (set by run_ensemble.py to .../member_K/).
    model_dir = args.output_dir or _THIS_DIR
    asyncio.run(run_training(HierSmartPlayer, args, model_dir))
