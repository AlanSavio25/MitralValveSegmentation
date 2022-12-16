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
        'trainable': True,
        'freeze_batch_normalization': False,
        'pretrained': True, # whether to use ImageNet weights,
        'optimizer': {
            'name': 'basic_optimizer',
        },
    }
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    strict_conf = False


    def _init(self, conf):

        self.conf = conf
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = smp.DeepLabV3Plus(in_channels=1, activation='sigmoid') # Model(pretrained=self.conf.pretrained)
        self.sem_seg_head = deepcopy(self.model.segmentation_head)
        self.box_seg_head = deepcopy(self.model.segmentation_head)
        self.model.segmentation_head = Identity()
        
    def _forward(self, data):
        image = data['image']
        encoding_decoding = self.model(image)
        box_seg = self.box_seg_head(encoding_decoding)
        sem_seg = self.sem_seg_head(encoding_decoding)
        pred = {'box_seg': box_seg, 'sem_seg': sem_seg}
        return pred
        
    def loss(self, pred, data):
        
        loss = {
        'total': 0.
        }

        
        loss_fn = smp.losses.JaccardLoss(mode='binary', from_logits=False)
        loss_box = loss_fn(pred['box_seg'], data['box'])
        loss['box'] = loss_box
        loss['total'] += loss_box
            
        return loss
        
    def metrics(self, pred, data):
        
#         metrics = {}
        
#         loss = nn.HuberLoss(reduction='sum')
#         l1_loss = nn.L1Loss(reduction='sum')
#         l2_loss = nn.MSELoss(reduction='sum')
        
#         # Roll metrics
#         if 'roll' in self.conf.heads:
#             gt_deg = (data['roll'].float()*(180./np.pi))
#             gt_norm = (gt_deg/45.) # normalized to [-1,1]
#             gt_class = torch.bucketize((data['roll'].float()*(180./np.pi)), self.roll_edges) - 1
#             if self.is_classification:
#                 output = pred['roll']
#                 pred_class = output.argmax(1)
#                 pred_deg = torch.tensor(self.roll_centers[pred_class], dtype=torch.float64, 
#                                   device=self.device)
#             else:
#                 pred_norm = pred['roll'].squeeze(1)
#                 pred_deg = pred_norm * 45.0
#                 pred_class = (torch.bucketize(pred_deg, self.roll_edges) - 1)
#                 print('roll met', pred_deg.shape, gt_deg.shape)
#                 assert pred_deg.dim() == gt_deg.dim()
#             print('roll met', pred_deg.shape, gt_deg.shape)
#             assert gt_deg.dim() == 1
#             metrics.update({
#                  'roll/Huber_degree_loss': torch.tensor([loss(pred_deg, gt_deg)]),
#                  'roll/L1_degree_loss': torch.tensor([l1_loss(pred_deg, gt_deg)]),                 
#                  'roll/L2_degree_loss': torch.tensor([l2_loss(pred_deg, gt_deg)]),
#             })
        
#         # Rho metrics: pitch degree L1 error, ratio errors
#         if 'rho' in self.conf.heads:
#             gt_ratio = (data['rho'].float())
#             gt_norm = (gt_ratio/0.35).unsqueeze(1) # normalized to [-1,1]
#             gt_class = torch.bucketize((gt_ratio/0.35), self.rho_edges) - 1
            
#             if self.is_classification:
#                 output = pred['rho']
#                 pred_class = output.argmax(1)
#                 pred_norm = torch.tensor(self.rho_centers[pred_class], dtype=torch.float64,
#                                   device=self.device)
#                 print(f'pred_norm: {pred_norm}')
#                 pred_norm = pred_norm.unsqueeze(0)
#             else:
#                 pred_norm = pred['rho'].squeeze(1)
#                 pred_class = (torch.bucketize(pred_norm, self.rho_edges) - 1)

#             pred_ratio = pred_norm * 0.35
            
#             # Compute pitch from predicted rho
#             H, W = data['height'].cpu().item(), data['width'].cpu().item()
#             F_px = data['f_px'].cpu().item()
#             f_ratio = data['focal_length_ratio_height'].cpu().item()
#             u0 = H / 2.
#             v0 = W / 2.
#             k1 = data['k1'].cpu().item()
#             k2 = data['k2'].cpu().item()
#             predicted_rho = pred_ratio.cpu().item()
#             rho_px = predicted_rho * H

#             img_pts = [u0, rho_px + v0]
#             camera = pycolmap.Camera(
#                     model='RADIAL',
#                     width=W,
#                     height=H,
#                     params=[F_px, u0, v0, k1, k2],
#                 )
#             normalized_coords = np.array(camera.image_to_world(img_pts))
#             camera_no_distortion = pycolmap.Camera(
#                     model='RADIAL',
#                     width=W,
#                     height=H,
#                     params=[F_px, u0, v0, 0.0, 0.0],
#                 )
#             back_to_image = np.array(camera_no_distortion.world_to_image(normalized_coords))
#             predicted_tau = (back_to_image[1] - v0) / H
            
#             predicted_pitch = np.arctan(predicted_tau/f_ratio)
            
#             gt_pitch_deg = data['pitch'].cpu().item() * 180/np.pi
#             pred_pitch_deg = predicted_pitch * 180/np.pi
#             print('rho met', pred_ratio, gt_ratio, pred_ratio.shape, gt_ratio.shape)
# #             print('rho met', pred_pitch_deg, gt_pitch_deg)
#             assert gt_ratio.dim() == 1
#             metrics.update({ 
#                  'rho/Huber_fraction_loss': torch.tensor([loss(pred_ratio, gt_ratio)]),
#                  'rho/L1_pitch_degree_loss': torch.tensor([l1_loss(torch.tensor(pred_pitch_deg), torch.tensor(gt_pitch_deg))])
#             })
            
#         # Field of View metrics    
#         if 'fov' in self.conf.heads:
#             gt_deg = (data['F_v'].float()*(180./np.pi))
#             h = 224
#             gt_pix = 1 / (torch.tan(gt_deg* (np.pi/180.) / 2) * 2 / h)
#             gt_class = torch.bucketize((data['F_v'].float()*(180./np.pi)), self.fov_edges) - 1
#             if self.is_classification:
#                 output = pred['fov']
#                 pred_class = output.argmax(1)
#                 pred_deg = torch.tensor(self.fov_centers[pred_class], dtype=torch.float64, device=self.device).unsqueeze(0)
#             else:
#                 pred_norm = pred['fov'].squeeze(1)
#                 min_fov = 55. 
#                 max_fov = 105.
#                 pred_deg = ((pred_norm + 1) * (max_fov - min_fov) / 2) + min_fov
#                 pred_class = (torch.bucketize(pred_deg, self.fov_edges) - 1)
#                 print('fov met', pred_deg.shape, gt_deg.shape)
#                 assert pred_deg.dim() == gt_deg.dim()
#             pred_pix = 1 / (torch.tan(pred_deg * (np.pi/180.) / 2) * 2 / h)
#             print('fov met', pred_deg.shape, gt_deg.shape)
#             print('fov met', pred_pix.shape, gt_pix.shape)


#             assert gt_deg.dim() == 1
#             assert pred_pix.dim() == gt_pix.dim() == 1
#             metrics.update({
#                  'fov/Huber_degree_loss': torch.tensor([loss(pred_deg, gt_deg)]),
#                  'fov/L1_degree_loss': torch.tensor([l1_loss(pred_deg, gt_deg)]),
#                  'fov/L1_pixel_loss': torch.tensor([l1_loss(pred_pix, gt_pix)]),
#             })
            
        metrics = {}
        
        return metrics