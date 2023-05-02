dataset_type='CocoDataset'
dataset_root='/mnt/extend3/datasets/coco2017'
classes=("unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
    "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports_ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk",
    "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "hair brush")

train_pipeline=dict(
    ReadImageFromFile=dict(color_type='color'),
    LoadAnnotations=dict(with_bbox=True),
    Resize=dict(scale=(768, 768), keep_ratio=True),
    RandomFlip=dict(prob=0.5),
)
test_pipeline=dict(
    ReadImageFromFile=dict(color_type='color'),
    LoadAnnotations=dict(with_bbox=True),
    # Resize=dict(scale_factor=1.0, keep_ratio=True)
)

train_dataloader=dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        image_dir='train2017',
        ann_file='annotations_trainval2017/annotations/instances_train2017.json',
        pipeline=train_pipeline,
    )
)
test_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        image_dir='val2017',
        ann_file='annotations_trainval2017/annotations/instances_val2017.json',
        pipeline=test_pipeline,
    )
)
val_dataloader=test_dataloader

# val_evaluator=dict(
#     type='CocoMetric',
#     ann_file='annotations_trainval2017/annotations/instances_val2017.json',
#     metric='bbox',
# )
# test_evaluator=val_evaluator
