# encoding: utf-8

import logging, os, sys, datetime
from .util_config import get_output_dir

class Logger:
    def __init__(self, cfg):
        self.logger_name = '-'.join([cfg.task.get('name'), 
                                     cfg.recorder.get('time_stamp')])
        self.logger = logging.getLogger(self.logger_name)
        self.logger.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        save_file_name = os.path.join(get_output_dir(cfg), "{}.log".format(
            self.logger_name.split('-')[0]))
        fh = logging.FileHandler(save_file_name, mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def get_logger(self):
        return self.logger

instance_logger = None

def setup_logger(cfg):
    global instance_logger
    if instance_logger:
        raise Exception("Failure to generate timestap!")
    else:
        generate_time_stamp(cfg)
        instance_logger = Logger(cfg).get_logger()
        return instance_logger

def get_current_logger(cfg=None):
    global instance_logger
    if instance_logger:
        return instance_logger
    elif cfg:
        return setup_logger(cfg)
    else:   
        raise Exception("Failure to get logger!")

def generate_time_stamp(cfg):
    cfg.recorder.update({'time_stamp': 
        datetime.datetime.now().strftime("%YY_%mM_%dD_%HH_%MM_%SS_%f")})
    