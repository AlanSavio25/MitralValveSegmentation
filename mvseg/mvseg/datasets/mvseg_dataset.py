"""
Dataset for training calibration network
"""


from pathlib import Path
from tqdm import tqdm
import numpy as np
import logging
import torch
import pickle
import json
from .base_dataset import BaseDataset
from ...settings import DATA_PATH, DATASETS_PATH
from .view import read_image, numpy_image_to_torch

import gzip
import os
import cv2
from copy import deepcopy
from skimage.transform import resize
logger = logging.getLogger(__name__)


class MVSegDataset(BaseDataset):
    default_conf = {
        'grayscale': True,
        'seed': 0,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        
        def load_zipped_pickle(filename):
            with gzip.open(filename, 'rb') as f:
                loaded_object = pickle.load(f)
            return loaded_object
        
        self.conf = conf
        self.split = split
        self.items = []
        if split == 'train':
            self.train_data = load_zipped_pickle(DATA_PATH + 'train.pkl')
            self.train_data.pop(45)
            self.train_data.pop(46)
            self.items = []
            for data in self.train_data:
                label = np.moveaxis(data['label'], -1, 0)
                video = np.moveaxis(data['video'], -1, 0)
                box = data['box'].astype('float32')
                for i, im in enumerate(video):
                    
                    l = label[i].astype('float32')
                    if not (im.shape[0] == im.shape[1] == 112):
                        im = resize(im, (112, 112), anti_aliasing=True)
                        box = resize(box, (112, 112), anti_aliasing=True)
                        l = resize(l, (112, 112), anti_aliasing=True)
                        
                    im = torch.from_numpy(im / 255.).float().unsqueeze(0)
                    box = box.astype('float32')
                    l = l.astype('float32')
                    item = {
                        'image': im,
                        'box': box,
                        'label': l
                    }
                    self.items.append(item)
            
        elif split == 'val':

            train_data = load_zipped_pickle(DATA_PATH + 'train.pkl')
            val_data = [train_data.pop(45), train_data.pop(46)]
            self.items = []
            for data in val_data:
                label = np.moveaxis(data['label'], -1, 0)
                video = np.moveaxis(data['video'], -1, 0)
                box = data['box'].astype('float32')
                for i, im in enumerate(video):
                    l = label[i].astype('float32')
                    
                    if not (im.shape[0] == im.shape[1] == 112):
                        im = resize(im, (112, 112), anti_aliasing=True)
                        box = resize(box, (112, 112), anti_aliasing=True)
                        l = resize(l, (112, 112), anti_aliasing=True)
                        
                    im = torch.from_numpy(im / 255.).float().unsqueeze(0)
                    box = box.astype('float32')
                    l = l.astype('float32')
                    
                    item = {
                        'image': im,
                        'box': box,
                        'label': l
                    }
                    self.items.append(item)
            
            
#         elif split == 'test':
#             self.test_data = load_zipped_pickle("test.pkl")

    def __getitem__(self, idx):
        data = self.items[idx]
        return data

    def __len__(self):
        return len(self.items)