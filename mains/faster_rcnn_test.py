# encoding: utf-8
import sys, argparse, math
sys.path.append('.')

from utils.util_config import *
from utils.util_logger import *
from utils.util_file import *
from utils.util_sys import *
from utils.util_visualizer import *

logger = None

def do_test(cfg, model, test_loader):
    device = 'cpu' if empty_config_node(cfg.task.get('devices')) else 'cuda'
    import torch

    model.eval()
    for idx, batch_data in enumerate(test_loader):
        images, targets = batch_data
        images = [image.to(device) for image in images]
        with torch.no_grad():
            batch_targets = model(images)
        for idx, target in enumerate(targets):
            infer_target = batch_targets[idx]
            draw_boxes(cfg, images[idx], targets[idx], infer_target)

def main():
    parser = argparse.ArgumentParser(description="Profile prefix")
    parser.add_argument("--config", default="config_detection", help="config file prefix", type=str)
    parser.add_argument("--weight", default="artefacts/weight.pt", help="model weight file", type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    global logger
    logger = setup_logger(cfg)
    logger.info('open this config file:\n{}'.format(config_to_str(cfg)))

    load_device_info(cfg)
    
    from utils.util_option import fixed_random_seed
    fixed_random_seed(cfg)

    from workers import plain_controller as controller
    cfg.model.update({'weight': args.weight})
    components = controller.load_tester_components(cfg)

    logger.info("The description of this task is: {}".format(cfg.task.get('version')))
    do_test(cfg, **components)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == '__main__':
    main()
