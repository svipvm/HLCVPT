# encoding: utf-8

import random, torch, numpy
from .util_logger import get_current_logger

def fixed_random_seed(cfg):
    seed = cfg.task.get('seed')
    if seed is None:
        seed = random.randint(1, 10000)
    logger = get_current_logger()
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

