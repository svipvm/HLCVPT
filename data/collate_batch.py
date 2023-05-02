# encoding: utf-8

import torch, numpy

def coco_collate_fn(batch):
    images = torch.cat([item['image'].unsqueeze(0) for item in batch], dim=0)
    target = [item['target'] for item in batch]
    # images - b x c x h x w
    # target - b x {key: value}
    return images, target

