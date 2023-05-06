# encoding: utf-8
import sys, argparse
sys.path.append('.')

from utils.util_config import *
from utils.util_logger import *
from utils.util_file import *
from utils.util_sys import *

import numpy as np

logger = None

def train_step(batch_data, model, optimizer, criticizer, device):
    images, targets = batch_data
    images = images.to(device)
    optimizer.zero_grad()
    boxes, losses = model(images, targets)
    total_loss = losses["loss_objectness"] + losses["loss_rpn_box_reg"]
    total_loss.backward()
    optimizer.step()

    loss_info = {}
    for key, loss in losses.items():
        loss_info[key] = loss.item()
    loss_info['loss'] = total_loss.item()

    return loss_info

def do_test(model, valid_loader, device):
    import torch
    loss_summer = {}
    model.eval()
    for batch_data in valid_loader:
        images, targets = batch_data
        images = images.to(device)
        with torch.no_grad():
            boxes, losses = model(images, targets)
        for key, loss in losses.items():
            if key not in loss_summer:
                loss_summer[key] = []
            loss_summer[key].append(loss)
    model.train()
    return {'mean_{}'.formar(key): np.mean(values) 
                for key, values in loss_summer.items()}


def main():
    parser = argparse.ArgumentParser(description="Profile prefix")
    parser.add_argument("--config", 
                default="config_detection", 
                help="config file prefix", 
                type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    global logger
    logger = setup_logger(cfg)
    logger.info('open this config file:\n{}'.format(config_to_str(cfg)))

    load_device_info(cfg)
    
    from utils.util_option import fixed_random_seed
    fixed_random_seed(cfg)

    from workers import plain_controller as controller
    components = controller.load_trainer_components(cfg)

    logger.info("The description of this task is: {}".format(cfg.task.get('version')))
    from workers.plain_trainer import do_train
    do_train(cfg, train_step, do_test, **components)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == '__main__':
    main()
