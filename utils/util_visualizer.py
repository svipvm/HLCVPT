import cv2, os, copy
import numpy as np
from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist

def __draw_boxes_for_one_image(image, target, classes):
    from torch import Tensor
    if isinstance(image, Tensor):
        image = image.cpu().detach().numpy()
        image = (np.transpose(image, (1, 2, 0)) * 255).astype(np.uint8)
    image = image.copy()
    boxes, labels, scores = target['boxes'], target['labels'], target.get('scores', None)
    with_score_flag = ('scores' in target)
    if with_score_flag:
        scores = target['scores'].cpu().detach().numpy()
    if not isinstance(boxes, Tensor):
        boxes, labels = boxes['data'], labels['data']
    for idx, bbox in enumerate(boxes):
        score = scores[idx] if with_score_flag else None
        try:
            bbox = bbox.cpu().detach().numpy()
        except:
            bbox = np.array(bbox)
        [left, top, right, bottom] = bbox[0:4].astype(int)
        # if score < 0.3:
        #     continue
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))
        if classes:
            fontScale = float(np.sqrt(right - left) / 16)
            text_str = classes[labels[idx]] + (' {}'.format(round(score, 2)) if score else '')
            cv2.putText(image, text_str, 
                        (left, bottom), 
                        cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1)
    return image


def draw_boxes(cfg, real_image, real_target, infer_target=None):
    show_dir = mkdir_if_not_exist([get_output_dir(cfg), 'show'])
    compare_flag = (infer_target and isinstance(infer_target, dict))
    # if compare_flag and (real_target['name'] != infer_target['name']):
    #     assert Exception("real_target and infer_target' name must be same!")
    image_file = real_target['name']
    classes = cfg.dataset_info.get('classes')
    real_anno_image = __draw_boxes_for_one_image(real_image, real_target, classes)
    if compare_flag != None:
        infer_anno_image = __draw_boxes_for_one_image(real_image, infer_target, classes)
        image = np.hstack([real_anno_image, infer_anno_image])
    else:
        image = real_anno_image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    save_name = os.path.basename(image_file).split('.')[0] + '.png'
    cv2.imwrite(os.path.join(show_dir, save_name), image)
        

