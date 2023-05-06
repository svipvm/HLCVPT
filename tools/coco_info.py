import sys, os
sys.path.append('.')

from utils.util_logger import *
from utils.util_file import *
from utils.util_config import *

from pycocotools.coco import COCO

# reference: https://leimao.github.io/blog/Inspecting-COCO-Dataset-Using-COCO-API
if __name__ == '__main__':
    cfg = load_config('config_detection')

    dataset_root = cfg.dataset_info.get('dataset_root')
    annotation_files = [
        os.path.join(dataset_root, cfg.train_dataloader.get('dataset').get('ann_file'))
    ]

    for annotation_file in annotation_files:
        coco_annotation = COCO(annotation_file=annotation_file)

        # Category IDs.
        cat_ids = coco_annotation.getCatIds()
        print(f"Number of Unique Categories: {len(cat_ids)}")
        print("Category IDs:")
        print(cat_ids)  # The IDs are not necessarily consecutive.

        # All categories.
        cats = coco_annotation.loadCats(cat_ids)
        cat_names = [cat["name"] for cat in cats]
        print("Categories Names:")
        print(cat_names)

        # Category ID -> Category Name.
        query_id = cat_ids[0]
        query_annotation = coco_annotation.loadCats([query_id])[0]
        query_name = query_annotation["name"]
        print(f"Category ID: {query_id}, Category Name: {query_name}")

        # Category Name -> Category ID.
        query_name = cat_names[2]
        query_id = coco_annotation.getCatIds(catNms=[query_name])[0]
        print(f"Category Name: {query_name}, Category ID: {query_id}")

        # Get the ID of all the images containing the object of the category.
        img_ids = coco_annotation.getImgIds(catIds=[query_id])
        print(f"Number of Images Containing {query_name}: {len(img_ids)}")
