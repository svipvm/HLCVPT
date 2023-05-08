optimizer=dict(
    type="SGD",
    params=dict(
        lr=0.00001, 
        momentum=0.90, 
        weight_decay=0.0001
    )
)

scheduler=dict(
    type='CosineAnnealingLR',
    params=dict(
        T_max=500,
        eta_min=0,
        last_epoch=-1,
        verbose=False
    )
)

criticizer=dict(
    type=[]
)

runner=dict(
    num_epochs=500,
    loss_period=300,
    eval_period=500,
    save_period=10000
)

recorder=dict(
    output_dir='work_dir',
    log_period=100
)

