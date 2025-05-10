import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RL-based ABR")
    parser.add_argument("--test", default=False, action="store_true", help="Evaluate only")
    parser.add_argument("--a2c", default=False, action="store_true", help="Train policy with A2C")
    parser.add_argument("--ppo", default=False, action="store_true", help="Train policy with PPO")
    parser.add_argument("--maml", default=False, action="store_true", help="Train policy with MAML")
    parser.add_argument("--lin", default=False, action="store_true", help="Using Linear metric")

    return parser
