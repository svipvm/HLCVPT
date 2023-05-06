# encoding: utf-8

import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *
from utils.util_visualizer import *
from data import build_dataloader

class TestDatalodaer(unittest.TestCase):

    def test_dataloader(self):
        cfg = load_config('config_detection')
        
        logger = setup_logger(cfg)
        logger.info('open this config file:\n{}'.format(config_to_str(cfg)))

        train_loader = build_dataloader(cfg, 2)
        d_iter = iter(train_loader)
        images, targets = next(d_iter)

        for idx in range(len(images)):
            logger.info('image shape:\n{}'.format(images[idx].shape))
            logger.info('target info:\n{}'.format(targets[idx]))
            draw_boxes(cfg, images[idx], targets[idx])

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
