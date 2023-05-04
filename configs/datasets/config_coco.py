dataset_type='CocoDataset'
dataset_root='/mnt/extend1/datasets/ShellFish'
classes=('shellfish', 'Crab', 'Lobster', 'Shrimp')

train_pipeline=dict(
    ReadImageFromFile=dict(color_type='color'),
    LoadAnnotations=dict(with_bbox=True),
    Resize=dict(scale=(416, 416), keep_ratio=True),
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
        image_dir='train',
        ann_file='train/_annotations.coco.json',
        pipeline=train_pipeline,
    )
)
test_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        image_dir='test',
        ann_file='test/_annotations.coco.json',
        pipeline=test_pipeline,
    )
)
val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=dataset_root,
        image_dir='valid',
        ann_file='valid/_annotations.coco.json',
        pipeline=test_pipeline,
    )
)

# val_evaluator=dict(
#     type='CocoMetric',
#     ann_file='annotations_trainval2017/annotations/instances_val2017.json',
#     metric='bbox',
# )
# test_evaluator=val_evaluator
