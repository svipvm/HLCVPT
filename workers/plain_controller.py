def load_trainer_components(cfg):
    from models import build_model
    model = build_model(cfg)

    from data import build_dataloader
    train_loader = build_dataloader(cfg, 0)
    valid_loader = build_dataloader(cfg, 1)

    from solvers import build_optimizer
    optimizer = build_optimizer(cfg, model)

    from solvers import build_scheduler
    scheduler = build_scheduler(cfg, optimizer)

    from functions import build_criticizer
    criticizer = build_criticizer(cfg)

    components = {
        'model': model,
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criticizer': criticizer
    }

    return components

def load_tester_components(cfg):
    from models import build_model
    model = build_model(cfg, False)

    from data import build_dataloader
    test_loader = build_dataloader(cfg, 2)

    components = {
        'model': model,
        'test_loader': test_loader
    }

    return components