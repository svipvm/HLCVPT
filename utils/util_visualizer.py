
import cv2, os
import numpy as np
from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist
from utils.util_data import COLOR_TYPE_MAP

def draw_bboex(cfg, image_info, predictions):
    logger = get_current_logger()

    show_dir = mkdir_if_not_exist([get_output_dir(cfg), 'show'])

    for index, prediction in enumerate(predictions):
        image_file = image_info[index]['name']
        image = cv2.imread(image_file, COLOR_TYPE_MAP[cfg.task.get('color_type')])
        
        for bbox in prediction:
            bbox = bbox.cpu().detach().numpy()
            [left, top, right, bottom] = bbox[0:4].astype(int)
            # if score < 0.3:
            #     continue
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))

        save_name = os.path.basename(image_file).split('.')[0] + '.png'
        des_file = os.path.join(show_dir, save_name)
        cv2.imwrite(os.path.join(show_dir, save_name), image)
        
    print('save file:', show_dir)

