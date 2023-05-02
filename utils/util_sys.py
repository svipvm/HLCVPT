# encoding: utf-8

import os
from .util_logger import get_current_logger

# ----------------------------------------
# GPU devices
# ----------------------------------------
def load_device_info(cfg):
    gpu_list = ','.join(str(x) for x in cfg.task.get('devices'))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    logger = get_current_logger()
    logger.info('CUDA_VISIBLE_DEVICES: ' + str(gpu_list))
