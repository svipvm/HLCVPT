# encoding: utf-8
import sys, argparse, math
sys.path.append('.')

from utils.util_config import *
from utils.util_logger import *
from utils.util_file import *
from utils.util_sys import *
from utils.util_visualizer import *

logger = None

def train_step(batch_data, model, optimizer, criticizer, device):
    images, targets = batch_data
    images = [image.to(device) for image in images]

    losses = model(images, targets)
    total_loss = sum(loss for loss in losses.values())

    if not math.isfinite(total_loss):
        logger.info("Loss is {}, stopping training".format(total_loss))
        sys.exit(1)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    loss_info = {}
    for key, loss in losses.items():
        loss_info[key] = loss.item()
    loss_info['loss'] = total_loss.item()

    return loss_info

def do_test(cfg, model, valid_loader, device):
    import torch
    model.eval()
    for idx, batch_data in enumerate(valid_loader):
        images, targets = batch_data
        images = [image.to(device) for image in images]
        with torch.no_grad():
            batch_targets = model(images, targets)
        for idx, target in enumerate(targets):
            infer_target = batch_targets[idx]
            # infer_target.update({'name': target['name']})
            draw_boxes(cfg, images[idx], targets[idx], infer_target)
        # for key, loss in losses.items():
        #     if key not in loss_summer:
        #         loss_summer[key] = []
        #     loss_summer[key].append(loss)
        if idx >= 35: break
    model.train()
    return {"info": "none"}

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
