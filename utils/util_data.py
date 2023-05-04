import numpy as np
import torch, os, random, cv2
from torchvision import transforms

COLOR_TYPE_MAP = {
    'color': cv2.IMREAD_COLOR,
    'grayscale': cv2.IMREAD_GRAYSCALE
}

DATA_TYPE_MAP = {
    'uint8': torch.uint8,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
}

def image2tensor(image):
    if len(image.shape) == 2: 
        image = np.expand_dims(image, 2)
    return transforms.ToTensor()(np.ascontiguousarray(image))

def target2tensor(target_):
    target = {}
    for key, data_dict in target_.items():
        if isinstance(data_dict, dict) and \
            'data' in data_dict and 'type' in data_dict:
            tensor = torch.from_numpy(np.ascontiguousarray(data_dict['data']))
            target[key] = tensor.type(DATA_TYPE_MAP[data_dict['type']])
        else:
            target[key] = data_dict
    return target

def __get_images_and_boxes(dataset_cfg):
    from pycocotools.coco import COCO
    ann_file = os.path.join(dataset_cfg.get('data_root'), dataset_cfg.get('ann_file'))
    coco = COCO(ann_file)
    img_ids = list(sorted(coco.imgs.keys()))
    dataset = {'shapes': [], 'labels': []}
    for image_id in img_ids:
        ann_list = coco.loadAnns(coco.getAnnIds(image_id))
        coco_img = coco.loadImgs(image_id)[0]
        # labels = []
        for ann in ann_list:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            # labels.append([category_id, x, y, w, h])
            dataset['shapes'].append([coco_img['width'], coco_img['height']])
            dataset['labels'].append([category_id, x, y, w, h])
    dataset['shapes'] = np.array(dataset['shapes'])
    dataset['labels'] = np.array(dataset['labels'])
    return dataset

