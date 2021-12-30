_base_ = '../cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'

############## HERE THE COPYPASTE STARTS ###############
# copypasted from mmdetection/configs/_base_/models/cascade_mask_rcn/workspaces/_fpn.py
# the only changed thing is the number of classes (changed in 3 lines)
NUM_CLASSES = 3
bbox_head=[
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=NUM_CLASSES,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                       loss_weight=1.0)),
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=NUM_CLASSES,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                       loss_weight=1.0)),
    dict(
        type='Shared2FCBBoxHead',
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=NUM_CLASSES,
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067]),
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
]
################### HERE THE COPYPASTE FINISHES #############

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=bbox_head,
        mask_head=dict(num_classes=3)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('astro', 'shsy5y', 'cort',)
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=18,
    train=dict(
        img_prefix='/data/kaggle_data/',
        classes=classes,
        ann_file='/data/mmdet/annotations_train.json'),
    val=dict(
        img_prefix='/data/kaggle_data/',
        classes=classes,
        ann_file='/data/mmdet/annotations_val.json'),
    )
checkpoint_config = dict(interval=6)