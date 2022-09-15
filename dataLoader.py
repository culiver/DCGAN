import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import glob
from torchvision import transforms
import numpy as np
from random import randint
from random import choice
import random
from PIL import Image
from tqdm import tqdm
import os 

class CelebA64_Dataset(Dataset):

    def __init__(self, config):
        self.mode = config['MODE']
        self.img_size = config['TRAINING_CONFIG']['RES']
        self.img_dir = os.path.join('data', 'CelebA', 'Img', 'img_align_celeba_64x64')
        self.sketch_dir = os.path.join('data', 'CelebA', 'Img', 'img_align_celeba_sketch')

        transform_list = []
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=(0.5), std=(0.5)))
        self.transform = transforms.Compose(transform_list)

        mode_dict = {'train':'0', 'val':'1', 'test':'2'}
        eval_txt = os.path.join('data', 'CelebA', 'Eval', 'list_eval_partition.txt')
        with open(eval_txt, 'r') as f:
            lines = f.readlines()
            self.data_list = [x.split()[0] for x in lines if x.split()[1].strip() == mode_dict[self.mode]]

    def __getitem__(self, index):
        fid = self.data_list[index]
        img = Image.open(os.path.join(self.img_dir, fid)).convert('RGB')
        img = self.transform(img)
        
        return img

    def __len__(self):
        """Return the number of images."""
        return len(self.data_list)


