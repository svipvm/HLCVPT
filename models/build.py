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

    logger.info("Model visible devices: \n{}".format(cfg.task.get('devices')))
    return DataParallel(model, cfg.task.get('devices'))

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

def __get_model(model_type, model_params, extra_params_dict=None):
    logger = get_current_logger()
    try:
        model = getattr(models, model_type)
        if isinstance(model_params, dict):
            if extra_params_dict:
                model = model(extra_params=extra_params_dict, **model_params)
            else:
                model = model(**model_params)
    except:
        message = "Failure to build this [{}] modle with {} parameters!".format(
                        model_type, model_params)
        logger.error(message)
        raise Exception(message)
    logger.info('{} model\n{}'.format(model_type, model))
    if extra_params_dict:
        no_requires_grad_model_list = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad: continue
            no_requires_grad_model_list.append(name)
        logger.info('No require gradient models: \n{}'.format(no_requires_grad_model_list))
    return model

def __parse_model(model_cfg: dict, main_model=True):
    if 'type' not in model_cfg or 'params' not in model_cfg: 
        return model_cfg, False
    if main_model: extra_params_dict = {}
    model_params_dict = {}
    if isinstance(model_cfg.get('params'), dict):
        for key, value in model_cfg.get('params').items():
            if isinstance(value, dict): 
                temp_value, extra_flag = __parse_model(value, False)
                if main_model and extra_flag and 'extra_params' in value:
                    extra_params_dict.update({key: value['extra_params']})
                value = temp_value
            model_params_dict[key] = value
    elif isinstance(model_cfg.get('params'), bool) and not model_cfg.get('params'):
        model_params_dict = False
    if main_model:
        model = __get_model(model_cfg.get('type'), model_params_dict, extra_params_dict)
        logger = get_current_logger()
        logger.info('{} model extra params\n{}'.format(
            model_cfg.get('type'), extra_params_dict))
        return model
    else:
        model = __get_model(model_cfg.get('type'), model_params_dict)
        return model, True

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
    