import json
import os
import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import jpeg4py as jpeg

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, 'data/emnist/images')
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/emnist')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')

def identity(x):
    return x

    
def get_transforms(size, backbone, s = 1):
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    
    if backbone == 'Conv4':
        normalization = transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                             np.array([0.229, 0.224, 0.225]))       
    elif backbone == 'Res12':
        normalization = transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                             np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
    elif backbone == 'Res18' or backbone == 'Res50':
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    else:
        raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
    
    data_transforms_aug = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.ToTensor(),
                                              normalization])
    
    data_transforms = transforms.Compose([transforms.Resize(size + 8),
                                          transforms.CenterCrop(size),
                                          transforms.ToTensor(),
                                          normalization])
    
    return data_transforms_aug, data_transforms


class Emnist(Dataset):
    def __init__(self, setname, args):
        set_mapper = {"train":"base", "val":"val", "test":"novel"}
        data_file = osp.join(SPLIT_PATH, set_mapper[setname] + '.json')
        with open(data_file, 'r') as f:
            self.meta = json.load(f)

        self.data, self.label = self.meta["image_names"], self.meta["image_labels"]
        self.num_class = len(set(self.label))

        image_size = 28
        # self.transform_aug, self.transform = get_transforms(image_size, args.backbone_class)
        self.transform_aug, self.transform = get_transforms(image_size, "Conv4")
        self.target_transform = identity

    def __getitem__(self,i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


if __name__ == "__main__":
    dataset = Emnist("val", None)
    print(dataset.label)