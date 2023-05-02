# encoding: utf-8

import torch
from torch.nn.parallel import DataParallel
from utils.util_logger import get_current_logger
from utils.util_config import empty_config_node
from torch.nn import init
import models.collector as models

def build_model(cfg, is_train=True):
    model_cfg = cfg.model
    device = 'cpu' if empty_config_node(cfg.task.get('devices')) else 'cuda'
    
    model = __parse_model(model_cfg)
    logger = get_current_logger()

    # pretrain field
    if is_train and not empty_config_node(model_cfg.get('pretrained')):
        load_model(model_cfg.get('pretrained'), model)
        logger.info('Pretrained weight: {}'.format(model_cfg.get('pretrained')))
    # test for artefact
    elif not is_train:
        # if empty_config_node(cfg.TEST.WEIGHT):
        #     raise Exception("Not found this weight!")
        # load_model(cfg.TEST.WEIGHT, model)
        # logger.info('Loading weight: {}'.format(cfg.TEST.WEIGHT))
        model.eval()
    # init model weight
    elif is_train:
        # init_type = cfg.MODELG.INIT_TYPE
        # init_bn_type = cfg.MODELG.INIT_BN_TYPE
        # gain = cfg.MODELG.INIT_GAIN
        # init_fn = functools.partial(__init_weight, init_type, init_bn_type, gain)
        # model.apply(init_fn)
        # logger.info('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(
        #     init_type, init_bn_type, gain))
        pass

    # logger.info("Model: \n{}".format(model))

    return DataParallel(model, cfg.task.get('devices')).to(device)

def save_model(model, path):
    if isinstance(model, DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key, param in state_dict.items():
        state_dict[key] = param.cpu()
    torch.save(state_dict, path)

def load_model(path, model, strict=True):
    if isinstance(model, DataParallel):
        model = model.module
    state_dict = torch.load(path)
    model.load_state_dict(state_dict, strict=strict)

def __get_model(model_type, model_params):
    logger = get_current_logger()
    try:
        model = getattr(models, model_type)
        model = model(**model_params)
    except:
        message = "Failure to build this [{}] modle with {} parameters!".format(
                        model_type, model_params)
        logger.error(message)
        raise Exception(message)
    logger.info('{} model: {}'.format(model_type, model))
    return model

def __parse_model(model_cfg: dict, main_model=True):
    if 'type' not in model_cfg or 'params' not in model_cfg: 
        return model_cfg
    model_params_dict = {}
    for key, value in model_cfg.get('params').items():
        if isinstance(value, dict): 
            value = __parse_model(value, False)
        model_params_dict[key] = value
    model = __get_model(model_cfg.get('type'), model_params_dict)
    return model

def __init_weight(model,  init_type, init_bn_type, gain):
    classname = model.__class__.__name__

    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        if init_type == 'normal':
                init.normal_(model.weight.data, 0, 0.1)
                model.weight.data.clamp_(-1, 1).mul_(gain)
        # add

        else:
            raise Exception("Not found this init type")

    elif classname.find('BatchNorm2d') != -1:
        if init_bn_type == 'uniform':  # preferred
                if model.affine:
                    init.uniform_(model.weight.data, 0.1, 1.0)
                    init.constant_(model.bias.data, 0.0)
        # add

        else:
            raise Exception("Not found this init bn type")
    