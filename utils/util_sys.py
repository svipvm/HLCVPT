# encoding: utf-8

import os
from .util_logger import get_current_logger

def load_device_info(cfg):
    gpu_list = ','.join(str(x) for x in cfg.task.get('devices'))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    logger = get_current_logger()
    logger.info('CUDA_VISIBLE_DEVICES: ' + str(gpu_list))

    import torch
    cfg.task.update({'devices': list(range(torch.cuda.device_count()))})
    gpu_list = ','.join(str(x) for x in cfg.task.get('devices'))
    logger.info('Torch visible gpu serial number: ' + str(gpu_list))
    