# encoding: utf-8

from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_file import mkdir_if_not_exist
from utils.util_config import empty_config_node
from utils.util_img import *
from models import save_model
from data.data_parser import do_parser
from .plain_tester import do_test
import torch

def do_train(cfg, train_step, do_test, model, train_loader, valid_loader, optimizer, scheduler, criticizer):
    device = 'cpu' if empty_config_node(cfg.task.get('devices')) else 'cuda'
    model_path = mkdir_if_not_exist([get_output_dir(cfg), 'model'])
    logger = get_current_logger()

    num_epochs = cfg.runner.get('num_epochs')
    log_period = cfg.recorder.get('log_period')
    test_period = cfg.recorder.get('test_period')
    save_period = cfg.recorder.get('save_period')
    trainer_step = 0

    logger.info('Begin of training.')
    for epoch in range(num_epochs):
        for batch_data in train_loader:
            trainer_step += 1

            train_step(batch_data, model, optimizer, criticizer, device)
            # images, targets = batch_data
            # images = images.to(device)
            # # targets = targets.to(device)
            # # y, x, hyper, _ = do_parser(cfg, batch_data)
            # optimizer.zero_grad()
            # # e = model(y) if not hyper else model(y, *hyper)
            # result = model(images, targets)
            # print(result)
            # l = criticizer(1, 1)
            # l.backward()
            # optimizer.step()

            # training information
            if trainer_step % log_period == 0:
                if scheduler:
                    current_lr = scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
                message = 'Train <epoch: {}/{}, iter: {}, lr: {:.3e}, loss: {:.3e}>.'.format(
                    epoch, num_epochs, trainer_step, current_lr, l.item())
                logger.info(message)

            # test model
            if trainer_step % test_period == 0:
                result = do_test(cfg, model, valid_loader)
                logger.info('Test <epoch: {}/{}, iter: {}> ' \
                            'result, average PSNR: {:<4.2f}dB, average SSIM: {:<4.2f}.'.format(
                            epoch, num_epochs, trainer_step, result['avg_psnr'], result['avg_ssim']))
            
            # save model
            if trainer_step % save_period == 0:
                save_model(model, os.path.join(model_path, str(trainer_step) + '.pt'))
                logger.info('Saving the model in step {}.'.format(trainer_step))

        # update leanring rate
        if scheduler:
            scheduler.step()

    result = do_test(cfg, model, valid_loader)
    logger.info('Test <final epoch, iter: {}> ' \
                'result, average PSNR: {:<4.2f}dB, average SSIM: {:<4.2f}.'.format(
                trainer_step, result['avg_psnr'], result['avg_ssim']))


    save_model(model, os.path.join(model_path, 'lastest.pt'))
    logger.info('End of training.')
