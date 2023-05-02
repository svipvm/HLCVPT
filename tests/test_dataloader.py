# encoding: utf-8

import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from data import build_dataloader

class TestDatalodaer(unittest.TestCase):

    def test_dataloader(self):
        cfg = load_config('config_detection')
        
        logger = setup_logger(cfg)
        logger.info('open this config file:\n{}'.format(config_to_str(cfg)))

        train_loader = build_dataloader(cfg, 0)
        d_iter = iter(train_loader)
        data_pair = next(d_iter)
        logger.info('image batch shape:\n{}'.format(data_pair[0].shape))
        logger.info('target batch shape:\n{}'.format(data_pair[1]))

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
