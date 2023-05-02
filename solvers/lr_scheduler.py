# encoding: utf-8

from torch.optim import lr_scheduler
from utils.util_logger import get_current_logger

def build(cfg, optimizer):
    scheduler_type = cfg.scheduler.get('type')
    
    try:
        scheduler = getattr(lr_scheduler, scheduler_type)
    except:
        raise Exception("Not found [{}] Scheduler!".format(scheduler_type))
    
    scheduler = scheduler(optimizer=optimizer, **cfg.scheduler.get('params'))

    logger = get_current_logger()
    logger.info("Scheduler: {}".format(scheduler))

    return scheduler