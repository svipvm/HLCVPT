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

        pipeline, train_loader = build_dataloader(cfg, 0)
        d_iter = iter(train_loader)
        data_pair = next(d_iter)
        logger.info('image batch shape:\n{}'.format(data_pair[0].shape))
        logger.info('target batch info:\n{}'.format(data_pair[1]))

        draw_info = data_pair[1][0]
        image, target = pipeline(draw_info['img_id_index'])
        logger.info('image batch shape:\n{}'.format(image.shape))
        logger.info('target batch info:\n{}'.format(target))

        draw_boxes(cfg, image, target)

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
