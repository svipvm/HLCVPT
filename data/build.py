# encoding: utf-8

from torch.utils import data

from utils.util_logger import get_current_logger
from .datasets import *
from .collate_batch import *
from .pipelines import *

def build_dataloader(cfg, data_type):
    if data_type == TRAIN_DATASET_TYPE:
        dataloder_cfg = cfg.train_dataloader
        shuffle = True
        drop_last = True
    else:
        dataloder_cfg = cfg.test_dataloader
        shuffle = False
        drop_last = False
    dataset_cfg = dataloder_cfg.get('dataset')
 
    pipeline = __get_pipeline(dataset_cfg)
    dataset = __get_dataset(dataset_cfg, pipeline)
    collate_fn = __get_collate(dataset_cfg)

    data_loader = data.DataLoader(
        dataset=dataset, 
        batch_size=dataloder_cfg.get('batch_size'),
        shuffle=shuffle, 
        collate_fn=collate_fn,
        num_workers=dataloder_cfg.get('num_workers'),
        drop_last=drop_last,
        pin_memory=True)

    return pipeline, data_loader

def __get_pipeline(dataset_cfg):
    dataset_type = dataset_cfg.get('type')

    if dataset_type in ['CocoDataset']:
        pipeline = DetectorPipe(dataset_cfg)
    # add pipeline 

    else:
        raise Exception("Not found this pipeline!")

    logger = get_current_logger()
    logger.info("Pipeline: {}".format(pipeline))

    return pipeline


def __get_dataset(dataset_cfg, pipeline):
    dataset_type = dataset_cfg.get('type')

    if dataset_type == "CocoDataset":
        dataset = CocoDataset(dataset_cfg, pipeline)
    # add dataset

    else:
        raise Exception("Not found this dataset!")

    logger = get_current_logger()
    logger.info("Dataset: {}".format(dataset))

    return dataset

def __get_collate(dataset_cfg):
    dataset_type = dataset_cfg.get('type')

    coco_dataset_list = ['CocoDataset']
    
    if dataset_type in coco_dataset_list:
        return coco_collate_fn
    # add collate

    else: # default collate
        return None 