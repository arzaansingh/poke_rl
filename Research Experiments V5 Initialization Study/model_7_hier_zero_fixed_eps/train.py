"""Train Model 7: Hierarchical + Zero Init + Fixed Epsilon"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.train_common import run_training, parse_train_args
from model_3_hier_zero.player import HierZeroPlayer

if __name__ == "__main__":
    args = parse_train_args()
    model_dir = os.path.dirname(os.path.abspath(__file__))
    asyncio.run(run_training(HierZeroPlayer, args, model_dir))
