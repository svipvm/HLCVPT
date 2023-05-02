# encoding: utf-8

from utils.util_logger import get_current_logger
from utils.util_config import get_output_dir
from utils.util_config import empty_config_node
from utils.util_file import mkdir_if_not_exist
import torch

def do_test(cfg, model, data_loader):
    result = {
        'avg_psnr': .0,
        'avg_ssim': .0
    }
    logger = get_current_logger(cfg)
    device = 'cpu' if empty_config_node(cfg.TASK.DEVICES) else 'cuda'
    count = len(data_loader)

    result_path = mkdir_if_not_exist([get_output_dir(cfg), 'result'])

    for idx, batch_data in enumerate(data_loader):
        y, x, hyper, paths = do_parser(cfg, batch_data)
        image_name, ext = os.path.splitext(os.path.basename(paths[0][0]))
        
        model.eval()
        with torch.no_grad():
            e = model(y) if not hyper else model(y, *hyper)
        model.train()

        img_e = tensor2uint(e.detach().float().cpu())
        img_h = tensor2uint(x.detach().float().cpu())

        current_ssim = calculate_ssim(img_e, img_h)
        current_psnr = calculate_psnr(img_e, img_h)

        logger.info('Test image: {:>7s} | PSNR: {:<4.2f}dB, SSIM: {:<4.2f}.'.format(
            image_name, current_psnr, current_ssim))

        result['avg_psnr'] += current_psnr
        result['avg_ssim'] += current_ssim

        imsave(img_e, os.path.join(result_path, image_name + ext))

    result['avg_psnr'] = result['avg_psnr'] / count
    result['avg_ssim'] = result['avg_ssim'] / count

    return result
