import numpy as np
import torch, os, random
from torchvision import transforms

DATA_TYPE_MAP = {
    'uint8': torch.uint8,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
}

def image2tensor(image):
    if len(image.shape) == 2: 
        image = np.expand_dims(image, 2)
    return transforms.ToTensor()(np.ascontiguousarray(image))

def target2tensor(target_):
    target = {}
    for key, data_dict in target_.items():
        if 'data' in data_dict and 'type' in data_dict:
            tensor = torch.from_numpy(np.ascontiguousarray(data_dict['data']))
            target[key] = tensor.type(DATA_TYPE_MAP[data_dict['type']])
        else:
            target[key] = data_dict
    return target

def __get_images_and_boxes(dataset_cfg):
    from pycocotools.coco import COCO
    ann_file = os.path.join(dataset_cfg.get('data_root'), dataset_cfg.get('ann_file'))
    coco = COCO(ann_file)
    img_ids = list(sorted(coco.imgs.keys()))
    dataset = {'shapes': [], 'labels': []}
    for image_id in img_ids:
        ann_list = coco.loadAnns(coco.getAnnIds(image_id))
        coco_img = coco.loadImgs(image_id)[0]
        # labels = []
        for ann in ann_list:
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']
            # labels.append([category_id, x, y, w, h])
            dataset['shapes'].append([coco_img['width'], coco_img['height']])
            dataset['labels'].append([category_id, x, y, w, h])
    dataset['shapes'] = np.array(dataset['shapes'])
    dataset['labels'] = np.array(dataset['labels'])
    return dataset

def kmean_anchors(config='config_detection', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            config: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from utils.util_config import load_config
    from utils.util_logger import get_current_logger

    from scipy.cluster.vq import kmeans
    from tqdm import tqdm

    cfg = load_config(config)
    logger = get_current_logger()
    npr = np.random
    thr = 1 / thr
    PREFIX = 'AutoAnchor: '

    dataset_cfg = cfg.train_dataloader.get('dataset')
    dataset = __get_images_and_boxes(dataset_cfg)

    # if isinstance(config, str):  # *.yaml file
        # with open(dataset, errors='ignore') as f:
        #     data_dict = yaml.safe_load(f)  # model dict
        # from utils.dataloaders import LoadImagesAndLabels
        # # modify
        # try:
        #     dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
        # except:
        #     dataset = LoadImagesAndLabels(f"{data_dict['path']}/{data_dict['train']}", augment=True, rect=True)

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            logger.info(s[:-2])
        return k

    # Get label wh
    print(dataset.get('shapes').shape, dataset.get('labels').shape)
    shapes = img_size * dataset.get('shapes') / dataset.get('shapes').max(1, keepdims=True)
    print(shapes.shape)
    # for s, l in zip(shapes, dataset.get('labels')):
    #     print(l[:, 3:5], s)
    #     print(l[:, 3:5] * s)
    # print([l[:, 3:5] * s for s, l in zip(shapes, dataset.get('labels'))])
    wh0 = np.concatenate([l[3:5] * s for s, l in zip(shapes, dataset.get('labels'))])  # wh

    # Filter
    i = (wh0 < 3.0).any().sum()
    if i:
        logger.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any()].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        logger.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...')
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        logger.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen))  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
