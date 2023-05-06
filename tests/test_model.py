# encoding: utf-8
import sys, unittest
sys.path.append('.')

from utils.util_logger import *
from utils.util_config import *
from utils.util_file import *
from utils.util_sys import *
from utils.util_visualizer import *

class TestMethod(unittest.TestCase):

    def test_method(self):
        cfg = load_config('config_detection')        
        
        logger = setup_logger(cfg)

        load_device_info(cfg)

        from utils.util_option import fixed_random_seed
        fixed_random_seed(cfg)

        from models import build_model
        TRAIN = True
        model = build_model(cfg, TRAIN)
        # from models.collector import Pytorch_FasterRCNN, ResNet50_Weights, FasterRCNN_ResNet50_FPN_Weights
        # model = Pytorch_FasterRCNN(weights=FasterRCNN_ResNet50_FPN_Weights)
        # model = Pytorch_FasterRCNN(weights_backbone=ResNet50_Weights)
        # logger.info('' + str(model))

        from data import build_dataloader
        TRAIN = False
        test_loader = build_dataloader(cfg, 0 if TRAIN else 1)
        data_tier = iter(test_loader)
        images, targets = next(data_tier)
        if TRAIN:
            losses = model(images, targets)
        else:
            import torch
            model.eval()
            with torch.no_grad():
                infer_targets = model(images)
            model.train()

        for idx in range(len(images)):
            if TRAIN:
                pass
            else:
                logger.info('image shape:\n{}'.format(images[idx].shape))
                logger.info('target info:\n{}'.format(targets[idx]))
                logger.info('infer_target info:\n{}'.format(infer_targets[idx]))
                draw_boxes(cfg, images[idx], targets[idx], infer_targets[idx])

        # from IPython import embed;
        # embed()


if __name__ == '__main__':
    unittest.main()
