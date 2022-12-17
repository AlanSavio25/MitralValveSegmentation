"""
"""
import torch
import torchvision
from .base_model import BaseModel
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn import Identity
import segmentation_models_pytorch as smp

class DeepLabV3(BaseModel):

    default_conf = {
        'name': 'deeplabv3',
        'finetune': False,
        'encoder': 'resnet101',
        'trainable': True,
        'freeze_batch_normalization': False,
        'pretrained': True, # whether to use ImageNet weights,
        'optimizer': {
            'name': 'basic_optimizer',
        },
    }
    
    strict_conf = False


    def _init(self, conf):

        self.conf = conf
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = smp.DeepLabV3Plus(encoder_name=self.conf.encoder, in_channels=1, activation='sigmoid')
#         self.sem_seg_head = deepcopy(self.model.segmentation_head)
        self.box_seg_head = deepcopy(self.model.segmentation_head)
        self.model.segmentation_head = Identity()
        
    def _forward(self, data):
        image = data['image']
        encoding_decoding = self.model(image)
        seg = self.box_seg_head(encoding_decoding)
#         sem_seg = self.sem_seg_head(encoding_decoding)
        pred = {'seg': seg} #, 'sem_seg': sem_seg}
        return pred
        
    def loss(self, pred, data):
        
        loss = {
        'total': 0.
        }

        loss_fn = smp.losses.JaccardLoss(mode='binary', from_logits=False).to(device=self.device)
        if self.conf.finetune:
            target = data['label']
        else:
            target = data['box']
        
        loss_box = loss_fn(pred['box_seg'], target).to(device=self.device) # for fine-tuning I changed 'box' to 'label'
        loss['box'] = loss_box
        loss['total'] += loss_box
        
#         loss_sem = loss_fn(pred['box_seg'], data['label']).to(device=self.device)
#         loss['sem'] = loss_sem
#         loss['total'] += loss_sem
        return loss
        
    def metrics(self, pred, data):
            
        metrics = {}
        
        return metrics