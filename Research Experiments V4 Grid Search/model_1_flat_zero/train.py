"""Train Model 1: Flat + Zero Init"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.train_common import run_training, parse_train_args
from model_1_flat_zero.player import FlatZeroPlayer

if __name__ == "__main__":
    args = parse_train_args()
    model_dir = os.path.dirname(os.path.abspath(__file__))
    asyncio.run(run_training(FlatZeroPlayer, args, model_dir))
