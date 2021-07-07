import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import warnings
import torchvision.transforms as transforms
from collections import OrderedDict

from torch.autograd import Variable
from model import *
from utils import *

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class SiamFCTracker:
    def __init__(self, model_path, gpu_id):
        self.gpu_id = gpu_id
        with torch.cuda.device(gpu_id):
            self.model = SiamFC()
            self.model.load_state_dict(copyStateDict(torch.load(model_path)))
            self.model = self.model.cuda()
            self.model.eval() 
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.exem_features = None
        self.scales = np.array([1.0255**(-2), 1.0255**(-1), 1.0255**(0), 1.0255**(1), 1.0255**(2)])
        self.dampling = 0.35
        self.cosine_weight = 0.3
        self.penalty = np.ones((5)) * 0.962
        self.penalty[5//2] = 1

        # create cosine window
        self.interp_response_sz = 16 * 17
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))
        
    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, bbox):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1+bbox[2], bbox[1]-1+bbox[3]) # zero based
        self.pos = np.array([bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])                            # width, height
        
        
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(img=frame, bbox=self.bbox, size_z=127, context_amount=0.5, img_mean=self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:]
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.exem_features = self.model.init_template(exemplar_img_var)
        

        # create s_x
        self.s_x = s_z + (255-127) / scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x
        
        
        return self.bbox
        
    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.pos, 255, size_x_scales, self.img_mean)
        instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0)
        
        
        with torch.cuda.device(self.gpu_id):
            instance_imgs_var = Variable(instance_imgs.cuda())
            response_maps = self.model.forward_corr(self.exem_features, instance_imgs_var)
            
            response_maps = response_maps.data.cpu().numpy().squeeze()
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC) for x in response_maps]
            
        
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty
        
        
        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        
        response_map = (1 - self.cosine_weight) * response_map + self.cosine_weight * self.cosine_window
        
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        
        # displacement in input
        disp_response_input = disp_response_interp * 8 / 16
        
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input * (self.s_x * scale) / 255
        
        # position in frame coordinates
        self.pos += disp_response_frame
        
       
        # scale damping and saturation
        self.s_x *= ((1 - self.dampling) + self.dampling * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - self.dampling) + self.dampling * scale) * self.target_sz
        
        
        bbox = (self.pos[0] - self.target_sz[0]/2+1, # xmin   convert to 1-based
                self.pos[1] - self.target_sz[1]/2+1, # ymin
                self.pos[0] + self.target_sz[0]/2+1, # xmax
                self.pos[1] + self.target_sz[1]/2+1) # ymax
        return bbox
    
    