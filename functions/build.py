# encoding: utf-8

from torch import nn
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node

def build_criticizer(cfg):
    device = 'cpu' if empty_config_node(cfg.task.get('devices')) else 'cuda'
    criticizer_types = cfg.criticizer.get('type')

    criticizer = []
    for loss_type in criticizer_types:
        if loss_type == "l1":
            criticizer.append(nn.L1Loss().to(device))
        elif loss_type == 'l2':
            criticizer.append(nn.MSELoss().to(device))
        # add loss function

        else:
            raise Exception("Not found this functions!")

    logger = get_current_logger()
    logger.info('Loss function: {}'.format(criticizer_types))

    return criticizer