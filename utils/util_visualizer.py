
import cv2, os
import numpy as np
from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist

def draw_bboex(cfg, images, image_info, predictions):
    logger = get_current_logger()

    show_dir = mkdir_if_not_exist([get_output_dir(cfg), 'show'])

    for index, prediction in enumerate(predictions):
        image = images[index].cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        print(image, image.shape)
        name = image_info[index]['name']
        for bbox, label_id, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            bbox = bbox.cpu().detach().numpy()
            [left, top, right, bottom] = bbox[0:4].astype(int)
            if score < 0.3:
                continue

            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))
        des_file = os.path.join(show_dir, name)
        print('save file:', des_file)
        cv2.imwrite(os.path.join(show_dir, name.split('.')[0] + '.png'), image)
        print('save file(finish):', os.path.join(show_dir, name.split('.')[0] + '.png'))

