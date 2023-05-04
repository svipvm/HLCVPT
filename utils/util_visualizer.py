
import cv2, os
import numpy as np
from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist
from utils.util_data import COLOR_TYPE_MAP

def draw_boxes(cfg, image, target):
    show_dir = mkdir_if_not_exist([get_output_dir(cfg), 'show'])
    classes = cfg.classes
    image_file = target['name']
    boxes, labels = target['boxes'], target['labels']
    if 'data' in boxes:
        boxes = boxes['data']
        labels = labels['data']
    for idx, bbox in enumerate(boxes):
        try:
            bbox = bbox.cpu().detach().numpy()
        except:
            bbox = np.array(bbox)
        [left, top, right, bottom] = bbox[0:4].astype(int)
        # if score < 0.3:
        #     continue
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))
        if classes:
            cv2.putText(image, classes[labels[idx]], 
                        (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    save_name = os.path.basename(image_file).split('.')[0] + '.png'
    # des_file = os.path.join(show_dir, save_name)
    cv2.imwrite(os.path.join(show_dir, save_name), image)
        

