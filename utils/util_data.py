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


