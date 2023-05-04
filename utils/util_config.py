# encoding: utf-8

import json
from .util_file import *
from importlib import import_module

def load_config(config_file):
    cfg = import_module(name='configs.{}'.format(config_file))
    return cfg

def config_to_str(cfg):
    return json.dumps(
        {key: getattr(cfg, key) for key in dir(cfg) if not key.startswith('_')},
        indent=4, sort_keys=False)

def get_output_dir(cfg):
    if cfg.recorder.get('time_stamp') is None:
        raise Exception("To generate timestap, please!")
    return mkdir_if_not_exist([
        cfg.recorder.get('output_dir'), 
        cfg.task.get('name'),
        cfg.recorder.get('time_stamp')
    ])

def empty_config_node(cnode):
    return not cnode
