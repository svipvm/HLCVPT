# encoding: utf-8
import sys, argparse
sys.path.append('.')

from utils.util_config import *
from utils.util_logger import *
from utils.util_file import *
from utils.util_sys import *
from workers import plain_controller as controller
from workers.plain_trainer import do_train

def train_step(batch_data, model, optimizer, criticizer, device):
    images, targets = batch_data
    images = images.to(device)
    # targets = targets.to(device)
    # y, x, hyper, _ = do_parser(cfg, batch_data)
    optimizer.zero_grad()
    # e = model(y) if not hyper else model(y, *hyper)
    result = model(images, targets)
    print(result)
    l = criticizer(1, 1)
    l.backward()
    optimizer.step()

def do_test():
    pass

def main():
    parser = argparse.ArgumentParser(description="Profile prefix")
    parser.add_argument("--config", 
                default="config_detection", 
                help="config file prefix", 
                type=str)
    args = parser.parse_args()

    cfg = load_config(args.config)

    logger = setup_logger(cfg)
    logger.info('open this config file:\n{}'.format(config_to_str(cfg)))

    load_device_info(cfg)
    
    from utils.util_option import fixed_random_seed
    fixed_random_seed(cfg)

    components = controller.load_components(cfg)
    logger.info("The description of this task is: {}".format(cfg.task.get('version')))
    do_train(cfg, train_step, do_test, **components)
    logger.info("This result was saved to: {}".format(get_output_dir(cfg)))

if __name__ == '__main__':
    main()
