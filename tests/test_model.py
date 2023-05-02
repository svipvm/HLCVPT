# encoding: utf-8
import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_config import *
from utils.util_file import *
from utils.util_sys import *

class TestMethod(unittest.TestCase):

    def test_method(self):
        cfg = load_config('config_detection')        
        
        logger = setup_logger(cfg)

        load_device_info(cfg)

        from utils.util_option import fixed_random_seed
        fixed_random_seed(cfg)

        from models import build_model
        TRAIN = False
        model = build_model(cfg, TRAIN)
        # logger.info('' + str(model))

        from data import build_dataloader
        test_loader = build_dataloader(cfg, 2)
        data_tier = iter(test_loader)
        batch = next(data_tier)
        predictions = model(batch[0])
        print(predictions)

        if not TRAIN:
            from utils.util_visualizer import draw_bboex
            draw_bboex(cfg, batch[0], batch[1], predictions)

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
