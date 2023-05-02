# encoding: utf-8

import torch.optim as optim

from utils.util_logger import get_current_logger


def build_optimizer(cfg, model):
    optimizer_type = cfg.optimizer.get('type')
    
    try:
        optimizer = getattr(optim, optimizer_type)
    except:
        raise Exception("Not found {} optimizer!".format(optimizer_type))
        
    params = []
    for key, param in model.named_parameters():
        if not param.requires_grad: continue
        params.append(param)
    
    optimizer = optimizer(params=params, **cfg.optimizer.get('params'))
    
    logger = get_current_logger()
    logger.info("Optimizer: {}".format(optimizer))

    return optimizer

