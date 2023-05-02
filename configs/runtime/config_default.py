optimizer=dict(
    type="SGD",
    params=dict(
        lr=0.0001, 
        momentum=0.90, 
        weight_decay=0.0001
    )
)

scheduler=dict(
    type='MultiStepLR',
    params=dict(
        milestones='linear',
        gamma=0.95
    )
)

criticizer=dict(
    type=['l2']
)

runner=dict(
    num_epochs=20,
    loss_period=300,
    eval_period=500,
    save_period=30000
)

recorder=dict(
    output_dir='work_dir',
    log_period=100
)

