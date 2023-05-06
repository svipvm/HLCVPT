dataset_info=dict(
    dataset_type='CocoDataset',
    dataset_root='/mnt/extend3/datasets/coco2017/',
    classes=("__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
             "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
             "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", 
             "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
             "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", 
             "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", 
             "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", 
             "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")
)

train_dataloader=dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_info['dataset_type'],
        data_root=dataset_info['dataset_root'],
        image_dir='train2017',
        ann_file='annotations_trainval2017/annotations/instances_train2017.json',
        pipeline=dict(
            ReadImageFromFile=dict(color_type='color'),
            LoadAnnotations=dict(with_bbox=True),
            # Resize=dict(scale=(416, 416), keep_ratio=True),
            RandomFlip=dict(prob=0.5),
        ),
    )
)

test_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_info['dataset_type'],
        data_root=dataset_info['dataset_root'],
        image_dir='val2017',
        ann_file='annotations_trainval2017/annotations/instances_val2017.json',
        pipeline=dict(
            ReadImageFromFile=dict(color_type='color'),
            LoadAnnotations=dict(with_bbox=True),
            # Resize=dict(scale_factor=1.0, keep_ratio=True)
            # Resize=dict(scale=(416, 416), keep_ratio=True),
        ),
    )
)

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_info['dataset_type'],
        data_root=dataset_info['dataset_root'],
        image_dir='val2017',
        ann_file='annotations_trainval2017/annotations/instances_val2017.json',
        pipeline=dict(
            ReadImageFromFile=dict(color_type='color'),
            LoadAnnotations=dict(with_bbox=True),
            # Resize=dict(scale_factor=1.0, keep_ratio=True)
            Resize=dict(scale=(416, 416), keep_ratio=True),
        ),
    )
)

# val_evaluator=dict(
#     type='CocoMetric',
#     ann_file='annotations_trainval2017/annotations/instances_val2017.json',
#     metric='bbox',
# )
# test_evaluator=val_evaluator
