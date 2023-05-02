# encoding: utf-8
import sys, unittest
sys.path.append('.')

import torch
from torchsummary import summary

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from data import build_dataloader
from models import build_model
from solvers import build_optimizer
from solvers import build_scheduler
from functions import build_criticizer


class TestSolver(unittest.TestCase):

    def test_solver(self):
        cfg = load_config('config_detection')

        logger = setup_logger(cfg)
        logger.info('open this config file:\n{}'.format(config_to_str(cfg)))
        
        model = build_model(cfg)
        optimzier = build_optimizer(cfg, model)
        logger.info("" + str(optimzier))

        scheduler = build_scheduler(cfg, optimzier)

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
