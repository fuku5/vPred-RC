import numpy as np
import torch
import torchvision
import functools

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import const

# DATASETS
# ADD DIR PATH HERE
DATA_DIRS = list()
DATA_DIRS.append(('./datasets/captcha-images/train', './datasets/captcha-images/test'))
DATA_DIRS.append(('./datasets/captcha-09az/train', './datasets/captcha-09az/val'))
DATA_DIRS.append(('./datasets/capitalized/train', './datasets/capitalized/test'))
DATA_DIRS.append(('./datasets/capital-color/train', './datasets/capital-color/test'))
DATA_DIRS = {Path(p).parent.name: [p, q] for p, q in DATA_DIRS}

#def get_data_names():
#    return [Path(p).parent.name for p, _ in DATA_DIRS]

class Mydataset(Dataset):

    def __init__(self, dir_path, transform=None, n=0):
        if type(dir_path) == str:
            if n > 0:
                img_paths = list(Path(dir_path).glob('*'))[:n]
            else:
                img_paths = list(Path(dir_path).glob('*'))
        elif hasattr(dir_path, '__iter__'):
            if n > 0:
                img_paths = sum([list(Path(p).glob('*'))[:n] for p in dir_path], list())
            else:
                img_paths = sum([list(Path(p).glob('*')) for p in dir_path], list())

        else:
            raise TypeError
        #self.img = [Image.open(path).convert('L') for path in img_paths]
        self.img = [Image.open(path).convert('RGB') for path in img_paths]
        self.labels = [path.stem for path in img_paths]

        self.transform = transform
        
    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.labels[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        return img, self.encode(label), label
    
    def __len__(self):
        return len(self.img)

    def encode(self, s):
        return torch.tensor([const.ALL_CHAR_SET.index(c) for c in s])


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    torchvision.transforms.ToTensor(),
])
transform_augment = torchvision.transforms.Compose([
    torchvision.transforms.Resize([224, 224]),
    #torchvision.transforms.RandomRotation(degrees=(0, 30)),
    torchvision.transforms.RandomAffine(degrees=(0, 5), translate=(0, 0), scale=(0.8, 1.0)),
    torchvision.transforms.ToTensor(),
])

@functools.lru_cache
def get_dataloader(names, batch_size, training=False, n=0):
    assert type(names) == tuple
    tf = transform_augment if training else transform
    dir_path = [DATA_DIRS[name][0] if training else DATA_DIRS[name][1] for name in names]
    dataset = Mydataset(dir_path, tf, n)

    shuffle =  training
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=shuffle)
    return dataloader

def load_datasets():
    dataloaders = dict()

    #for i, (train_dir, test_dir) in enumerate(const.DATA_DIRS):
    for data_name, (train_dir, test_dir) in get_data_names_and_paths.items():

        train_dl = get_dataloader(train_dir, 64, True)
        test_dl = get_dataloader(test_dir, 256, False)
        dataloaders['train_dataloader_{}'.format(data_name)] = train_dl
        dataloaders['test_dataloader_{}'.format(data_name)] = test_dl
        
    TRAIN_DATA_DIR_ALL, TEST_DATA_DIR_ALL = zip(*DATA_DIRS)

    #dataloaders['train_dataloader_01'] = get_dataloader(TRAIN_DATA_DIR_ALL[:2], 64, True)
    #dataloaders['test_dataloader_01'] = get_dataloader(TEST_DATA_DIR_ALL[:2], 64, False)

    dataloaders['train_dataloader_all'] = get_dataloader(TRAIN_DATA_DIR_ALL, 64, True)
    dataloaders['test_dataloader_all'] = get_dataloader(TEST_DATA_DIR_ALL, 64, False)

    return dataloaders
