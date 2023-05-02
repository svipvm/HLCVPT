from pycocotools.coco import COCO
import os, cv2
import numpy as np
from utils.util_data import COLOR_TYPE_MAP

def __get_data_from_container(container, image_id, data_type):
    if isinstance(container, COCO):
        if data_type == 'image':
            return container.loadImgs(image_id)[0]["file_name"]
        elif data_type == 'annotations':
            return container.loadAnns(container.getAnnIds(image_id))

def __get_target_from_annotations(container, ann_list):
    target = {}
    if isinstance(container, COCO):
        boxes, label_ids = [], []
        for ann in ann_list:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            label_ids.append(ann['category_id'])
        # must to generate index ('data', 'type')
        target['boxes'] = {'data': boxes, 'type': 'fp16'}
        target['labels'] = {'data': label_ids, 'type': 'int64'}
    return target

def read_image_from_file(pipeline_cfg, data_cfg, img_id, cfg_index='ReadImageFromFile'):
    pipeline_cfg = pipeline_cfg.get(cfg_index)
    if not pipeline_cfg:
        raise Exception("The pipe must load a image!")
    image_path = __get_data_from_container(data_cfg.get('container'), img_id, 'image')
    image_file = os.path.join(data_cfg.get('data_dir'), image_path)
    target = dict(name=image_file)
    image = cv2.imread(image_file, COLOR_TYPE_MAP[pipeline_cfg.get('color_type')])
    return image, target

def load_annotations(pipeline_cfg, data_cfg, img_id, cfg_index='LoadAnnotations'):
    pipeline_cfg = pipeline_cfg.get(cfg_index)
    if not pipeline_cfg or not pipeline_cfg.get('with_bbox'):
        return None
    ann_list = __get_data_from_container(data_cfg.get('container'), img_id, 'annotations')
    target = __get_target_from_annotations(data_cfg.get('container'), ann_list)
    return target

def resize(pipeline_cfg, image, target, cfg_index='Resize'):
    pipeline_cfg = pipeline_cfg.get(cfg_index)
    if not pipeline_cfg:
        return image, target
    H, W = image.shape[:2]
    scale = pipeline_cfg.get('scale')
    h_rate, w_rate = scale[0] / H, scale[1] / W
    image = cv2.resize(image, (scale[1], scale[0]), interpolation=cv2.INTER_NEAREST)
    target['boxes']['data'] = [[data[0] * w_rate, data[1] * h_rate,
                                data[2] * w_rate, data[3] * h_rate]
                                for data in target['boxes']['data']]
    return image, target

def random_flip(pipeline_cfg, image, target, cfg_index='RandomFlip'):
    pipeline_cfg = pipeline_cfg.get(cfg_index)
    if not pipeline_cfg:
        return image, target
    if np.random.rand() < pipeline_cfg.get('prob'):
        return image, target
    return image, target