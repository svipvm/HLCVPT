# encoding: utf-8

import torch, numpy

def coco_collate_fn(batch):
    images = [item['image'] for item in batch]
    target = [item['target'] for item in batch]
    # images - b x c x h x w
    # target - b x {key: value}
    return images, target

