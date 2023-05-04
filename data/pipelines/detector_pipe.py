# encoding: utf-8

import os
from pycocotools.coco import COCO
from .base_pipelines import *

class DetectorPipe():
    def __init__(self, dataset_cfg):
        self.pipeline_cfg = dataset_cfg.get('pipeline')
        ann_file = os.path.join(dataset_cfg.get('data_root'), dataset_cfg.get('ann_file'))
        coco = COCO(ann_file)
        self.data_cfg = {
            'data_dir': os.path.join(dataset_cfg.get('data_root'), dataset_cfg.get('image_dir')),
            'ann_file': ann_file,
            'container': coco,
            'img_ids': list(sorted(coco.imgs.keys()))
        }

    def __call__(self, index):
        img_id = self.data_cfg.get('img_ids')[index]
        image, target = read_image_from_file(self.pipeline_cfg, self.data_cfg, img_id)
        target.update(load_annotations(self.pipeline_cfg, self.data_cfg, img_id))
        image, target = resize(self.pipeline_cfg, image, target)
        # image, target = random_flip(self.pipeline_cfg, image, target)
        target.update({'img_id_index': index})
        return image, target
    
    def __len__(self):
        return len(self.data_cfg.get('img_ids'))