from torch.utils.data import Dataset
import utils.util_data as data_util
import albumentations as T


class CocoDataset(Dataset):
    def __init__(self, dataset_cfg, pipeline):
        super(CocoDataset, self).__init__()
        self.dataset_cfg = dataset_cfg
        self.pipeline = pipeline(self.dataset_cfg)
        self.transforms = T.Compose([
            T.Normalize()
        ])

    def __getitem__(self, index):
        image, target = self.pipeline(index)
        image = self.transforms(image=image)['image']

        image = data_util.image2tensor(image)
        target = data_util.target2tensor(target)
        
        return {'image': image, 'target': target}

    def __len__(self):
        return len(self.pipeline)

