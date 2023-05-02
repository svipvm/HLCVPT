# encoding: utf-8

from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist
from utils.util_config import empty_config_node
from models import save_model
import torch, os
import numpy as np

def do_train(cfg, train_step, do_test, model, train_loader, valid_loader, optimizer, scheduler, criticizer):
    device = 'cpu' if empty_config_node(cfg.task.get('devices')) else 'cuda'
    model_path = mkdir_if_not_exist([get_output_dir(cfg), 'model'])
    logger = get_current_logger()

    log_period = cfg.recorder.get('log_period')
    num_epochs = cfg.runner.get('num_epochs')
    loss_period = cfg.runner.get('loss_period')
    eval_period = cfg.runner.get('eval_period')
    save_period = cfg.runner.get('save_period')

    trainer_step = 1
    loss_summer = {}

    logger.info('Begin of training.')
    for epoch in range(num_epochs):
        batch_size = len(train_loader)
        for batch_data in train_loader:
            losses = train_step(batch_data, model, optimizer, criticizer, device)

            for key, loss in losses.items():
                if key not in loss_summer:
                    loss_summer[key] = []
                loss_summer[key].append(loss)
            
            if trainer_step % log_period == 0:
                if scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                message = 'Train <epoch: {}/{}, iter: {}/{}, lr: {:.3e}>'.format(
                    epoch, num_epochs, trainer_step, batch_size, current_lr)
                for idx_, (key, loss) in enumerate(losses.items()):
                    message += ', {}: {:.3e}'.format(key, loss)
                logger.info(message)

            if trainer_step % loss_period == 0:
                message = 'Train <epoch: {}/{}, iter: {}/{}. mean loss'.format(
                    epoch, num_epochs, trainer_step, batch_size)
                for idx_, (key, loss) in enumerate(loss_summer.items()):
                    message += ', {}: {:.3e}'.format(key, np.mean(loss))
                logger.info(message)
                loss_summer = {}
            
            if trainer_step % eval_period == 0:
                result = do_test(model, valid_loader, device)
                message = 'Test <epoch: {}/{}, iter: {}/{}>. test info'.format(
                    epoch, num_epochs, trainer_step, batch_size)
                for idx_, (key, info) in enumerate(result.items()):
                    message += ', {}: {:.3e}'.format(key, np.mean(info))
                logger.info(message)
            
            # save model
            if trainer_step % save_period == 0:
                save_model(model, os.path.join(model_path, str(trainer_step) + '.pt'))
                logger.info('Saving the model in step {}.'.format(trainer_step))

            trainer_step += 1

        # update leanring rate
        if scheduler:
            scheduler.step()

    result = do_test(model, valid_loader, device)
    message = 'Test lastest. test info'.format(
        epoch, num_epochs, trainer_step, batch_size)
    for idx_, (key, info) in enumerate(result.items()):
        message += ', {}: {:.3e}'.format(key, np.mean(info))
    logger.info(message)

    save_model(model, os.path.join(model_path, 'lastest.pt'))
    logger.info('End of training.')
