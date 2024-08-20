import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np

from pathlib import Path
from runners.diffusion import Diffusion


arg_dict = {
    "config": "cifar10.yml",
    "seed": 1234,
    "exp": "./workdir/exp",
    "doc": "./workdir/doc",
    "log_path": "./workdir/log",
    "verbose": "info",
    "timesteps": 1000,
}


def parse_args_and_config():
    args = dict2namespace(arg_dict)

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    args.log_path = Path(args.log_path)
    args.log_path.mkdir(exist_ok=True, parents=True)

    with open(os.path.join(args.log_path, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    
    runner = Diffusion(args, config)
    runner.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())
