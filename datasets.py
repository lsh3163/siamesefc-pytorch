from torch.utils.data.dataset import Dataset

import os
from glob import glob
import cv2
import random
import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
from utils import *
from torchvision import transforms
import torch
import json
def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img
    
def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def centercrop(img, size):
    shape = img.shape[:2]
    cy, cx = shape[0] // 2, shape[1] // 2
    ymin, xmin = cy - size//2, cx - size// 2
    ymax, xmax = cy + size//2 + 1, cx + size//2 + 1
    left = right = top = bottom = 0
    im_h, im_w = shape
    if xmin < 0:
        left = int(abs(xmin))
    if xmax > im_w:
        right = int(xmax - im_w)
    if ymin < 0:
        top = int(abs(ymin))
    if ymax > im_h:
        bottom = int(ymax - im_h)

    xmin = int(max(0, xmin))
    xmax = int(min(im_w, xmax))
    ymin = int(max(0, ymin))
    ymax = int(min(im_h, ymax))
    im_patch = img[ymin:ymax, xmin:xmax]
    if left != 0 or right !=0 or top!=0 or bottom!=0:
            im_patch = cv2.copyMakeBorder(im_patch, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=0)
    return im_patch


def randomcrop(img, size):
    shape = img.shape[:2]
    y1 = np.random.randint(0, shape[0] - size + 1)
    x1 = np.random.randint(0, shape[1] - size + 1)
    y2 = max(y1 + size, shape[0])
    x2 = max(x1 + size, shape[1])
    img_patch = img[y1:y2, x1:x2]
    return img_patch

def shift(img, x, y):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, x], [0, 1, y]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

class Ilsvrc15Set(Dataset):
    def __init__(self, root_path="./datasets/ILSVRC2015", mode="train", viz=True):
        self.mode = mode
        self.video_paths_txt = glob(root_path + "/ImageSets/VID/{}*.txt".format(self.mode))
        self.video_paths, self.anno_paths = [], [] 
        self.transform = transforms.Compose(
            [

                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]) 
        
        self.viz = viz
        for video_path in self.video_paths_txt:
            f = open(video_path)
            lines = f.readlines()
            for line in lines:
                video_name = line.split()[0]
                single_video_path = root_path + "/Data/VID/" + self.mode + "/" + video_name
                single_anno_path = root_path + "/Annotations/VID/" + self.mode + "/" + video_name
                self.video_paths.append(single_video_path)
                self.anno_paths.append(single_anno_path)
        
        self.val_video_paths = root_path + "/ImageSets/VID/val.txt"
        f = open(self.val_video_paths)
        lines = f.readlines()
        for line in lines:
            video_name = line.split()[0]
            single_video_path = root_path + "/Data/VID/val/" + video_name
            single_anno_path = root_path + "/Annotations/VID/val/" + video_name
            # print(single_video_path[:59], single_anno_path[:66])
            self.video_paths.append(single_video_path[:59])
            self.anno_paths.append(single_anno_path[:66])
       
        
    def __getitem__(self, idx):
        images_list = sorted(glob(self.video_paths[idx]+"/*.JPEG"))
        annos_list = sorted(glob(self.anno_paths[idx]+"/*.xml"))
        length_images = len(images_list)
        
        exemplar_idx, search_idx = random.randint(0, length_images - 1), random.randint(0, length_images - 1)
        # search_idx = min(exemplar_idx + random.randint(1, 100), length_images - 1)
        
#         if np.random.rand(1) < 0.5:
#             exemplar_idx, search_idx = search_idx, exemplar_idx
            
        exem_image_path, srh_image_path = images_list[exemplar_idx], images_list[search_idx]
        exem_anno_path, srh_anno_path = annos_list[exemplar_idx], annos_list[search_idx]
        
        exem_img, srh_img = cv2.imread(exem_image_path), cv2.imread(srh_image_path)
        exem_img, srh_img = cv2.cvtColor(exem_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(srh_img, cv2.COLOR_BGR2RGB)
        # print(exem_image_path, exem_anno_path)
        
        
        exem_xmin, exem_ymin, exem_xmax, exem_ymax = 0, 0, exem_img.shape[0], exem_img.shape[1] 
        srh_xmin, srh_ymin, srh_xmax, srh_ymax = 0, 0, srh_img.shape[0], srh_img.shape[1] 
        
        
        ### Getting Bounding Box of dog (ROI) in Exemplar Image
        tree = ET.parse(exem_anno_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            exem_xmin = int(bndbox.find('xmin').text)
            exem_ymin = int(bndbox.find('ymin').text)
            exem_xmax = int(bndbox.find('xmax').text)
            exem_ymax = int(bndbox.find('ymax').text)
        
        tree = ET.parse(srh_anno_path)
        root = tree.getroot()
        objects = root.findall('object')
        for obj in objects:
            bndbox = obj.find('bndbox')
            srh_xmin = int(bndbox.find('xmin').text)
            srh_ymin = int(bndbox.find('ymin').text)
            srh_xmax = int(bndbox.find('xmax').text)
            srh_ymax = int(bndbox.find('ymax').text)
        
        srh_img, _, _ = get_instance_image(img=srh_img, bbox=(srh_xmin, srh_ymin, srh_xmax, srh_ymax), size_z=127, size_x=255, context_amount=0.5, img_mean=0)
        exem_img, _, _ = get_exemplar_image(img=exem_img, bbox=(exem_xmin, exem_ymin, exem_xmax, exem_ymax), size_z=127, context_amount=0.5, img_mean=0)
        
        scale_h, scale_w = 1.0 + np.random.uniform(-0.05, 0.05), 1.0 + np.random.uniform(-0.05, 0.05)
        
        exem_img = cv2.resize(exem_img, (int(scale_h * 127), int(scale_w * 127)), cv2.INTER_CUBIC)
        srh_img = cv2.resize(srh_img, (int(scale_h * 255), int(scale_w * 255)), cv2.INTER_CUBIC)
        
        exem_img = centercrop(exem_img, 127)

        
        if scale_h > 1.0 and scale_w > 1.0:
            srh_img = randomcrop(srh_img, 255)
            
        srh_img = centercrop(srh_img, 255)
#         else:
#             srh_img = centercrop(srh_img, 255)
        
        if np.random.rand(1) < 0.25:
            exem_img = cv2.cvtColor(exem_img, cv2.COLOR_RGB2GRAY)
            exem_img = cv2.cvtColor(exem_img, cv2.COLOR_GRAY2RGB)
            srh_img = cv2.cvtColor(srh_img, cv2.COLOR_RGB2GRAY)
            srh_img = cv2.cvtColor(srh_img, cv2.COLOR_GRAY2RGB)
            
#         if np.random.rand(1) < 0.25:
#             exem_img, srh_img = horizontal_flip(exem_img, True), horizontal_flip(srh_img, True)
            
#         if np.random.rand(1) < 0.25:
#             exem_img, srh_img = rotation(exem_img, 10), rotation(srh_img, 10)
            
#         if np.random.rand(1) < 0.25:
#             exem_img, srh_img = brightness(exem_img, 0.5, 3), brightness(srh_img, 0.5, 3)
            
#         if np.random.rand(1) < 0.05:
#             exem_img = cv2.GaussianBlur(exem_img, (5, 5), 0)
#             srh_img = cv2.GaussianBlur(srh_img, (5, 5), 0)
            
            
        if self.viz:
            show_exem_img, show_srh_img = cv2.cvtColor(exem_img, cv2.COLOR_RGB2BGR), cv2.cvtColor(srh_img, cv2.COLOR_RGB2BGR)
            os.makedirs("./outputs" + "/" +str(idx), exist_ok=True)
            cv2.imwrite("./outputs" + "/" +str(idx) + "/" + str(search_idx) + "srh.png", srh_img)
            cv2.imwrite("./outputs" + "/" +str(idx) + "/" + str(exemplar_idx) + "exem.png", exem_img)
            
        exem_img, srh_img = self.transform(exem_img), self.transform(srh_img)
        
        
        
        h, w = 17, 17
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
        mask[dist <= 16 / 8] = 1
        mask[dist > 16 / 8] = -1
        
        mask = mask[np.newaxis, :, :]
        
        score_gt = torch.from_numpy(mask)
        
        weights = np.ones_like(mask)
        
        # weights = np.exp(-((dist)**2 / (2.0*1)))
        pos_num = np.sum(mask == 1)
        neg_num = np.sum(mask == 0)
        
        weights[mask == 1] = 0.5 / pos_num
        weights[mask == 0] = 0.5 / neg_num
        
        # weights *= pos_num + neg_num
        weights = torch.from_numpy(weights)
        return exem_img, srh_img, score_gt, weights
    
    
    def __len__(self):
        return len(self.video_paths)

    
    
    
class VID(Dataset):

    def __init__(self, imdb, data_dir, phase='train'):
        imdb_video      = json.load(open(imdb, 'r'))
        self.videos     = imdb_video['videos']
        self.data_dir   = data_dir
        self.num_videos = int(imdb_video['num_videos'])

        self._imgpath = os.path.join(self.data_dir, "%s", "%06d.%02d.x.jpg")
        self.transform = transforms.Compose(
            [

                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]) 

        assert(phase in ['train', 'val'])
        if phase == 'train':
            self.itemCnt = 53200
        else:
            self.itemCnt = self.num_videos

    def __getitem__(self, idx):
        idx = idx % self.num_videos
        video = self.videos[idx]
        trajs = video['trajs']
        # sample one trajs
        trackid = np.random.choice(list(trajs.keys()))
        traj = trajs[trackid]

        rand_z = np.random.choice(range(len(traj)))
        possible_x_pos = list(range(len(traj)))
        rand_x = np.random.choice(possible_x_pos[max(rand_z - 200, 0):rand_z] + possible_x_pos[(rand_z + 1):min(rand_z + 200, len(traj))])

        z = traj[rand_z].copy()
        x = traj[rand_x].copy()

        # read z and x
        img_z = cv2.imread(self._imgpath % (video['name'], int(z['fno']), int(trackid)))
        img_z = cv2.cvtColor(img_z, cv2.COLOR_BGR2RGB)

        img_x = cv2.imread(self._imgpath % (video['name'], int(x['fno']), int(trackid)))
        img_x = cv2.cvtColor(img_x, cv2.COLOR_BGR2RGB)
        
        

        scale_h, scale_w = 1.0 + np.random.uniform(-0.05, 0.05), 1.0 + np.random.uniform(-0.05, 0.05)
        img_x = cv2.resize(img_x, (int(scale_h * 255), int(scale_w * 255)), cv2.INTER_CUBIC)
        scale_h, scale_w = 1.0 + np.random.uniform(-0.05, 0.05), 1.0 + np.random.uniform(-0.05, 0.05)
        img_z = cv2.resize(img_z, (int(scale_h * 255), int(scale_w * 255)), cv2.INTER_CUBIC)
        
        
        
        img_z = centercrop(img_z, 127)
        img_x = centercrop(img_x, 255)
        
        if np.random.rand(1) < 0.25:
            img_z = cv2.cvtColor(img_z, cv2.COLOR_RGB2GRAY)
            img_z  = cv2.cvtColor(img_z, cv2.COLOR_GRAY2RGB)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_RGB2GRAY)
            img_x = cv2.cvtColor(img_x, cv2.COLOR_GRAY2RGB)
            
#         if np.random.rand(1) < 0.25:
#             img_x, img_z = horizontal_flip(img_x, True), horizontal_flip(img_z, True)
            
#         if np.random.rand(1) < 0.25:
#             img_x, img_z = rotation(img_x, 10), rotation(img_z, 10)
            
#         if np.random.rand(1) < 0.25:
#             img_x, img_z = brightness(img_x, 0.5, 2), brightness(img_z, 0.5, 2)
            
#         if np.random.rand(1) < 0.05:
#             img_x = cv2.GaussianBlur(img_x, (5, 5), 0)
#             img_z = cv2.GaussianBlur(img_z, (5, 5), 0)
            
        if np.random.rand(1) < 0.25:
            img_x = shift(img_x, np.random.randint(-10, 10), np.random.randint(-10, 10))
            img_z = shift(img_z, np.random.randint(-10, 10), np.random.randint(-10, 10))
        
#         img_z, img_x = img_z.transpose(2, 0, 1), img_x.transpose(2, 0, 1)
#         # img_z, img_x = torch.from_numpy(img_z.astype(np.float32)), torch.from_numpy(img_x.astype(np.float32))
        
        
        h, w = 17, 17
        y = np.arange(h, dtype=np.float32) - (h-1) / 2.
        x = np.arange(w, dtype=np.float32) - (w-1) / 2.
        y, x = np.meshgrid(y, x)
        dist = np.sqrt(x**2 + y**2)
        mask = np.zeros((h, w))
#         mask[dist <= 16 / 8] = 1
#         mask[dist > 16 / 8] = -1
        mask[dist <= 1] = 1
        mask[dist > 1] = -1
 
        mask = mask[np.newaxis, :, :]
        
        score_gt = torch.from_numpy(mask)
        
        weights = np.ones_like(mask)
        
        pos_num = np.sum(mask == 1)
        neg_num = np.sum(mask == -1)
        
        weights[mask == 1] = 0.5 / pos_num
        weights[mask == -1] = 0.5 / neg_num
        
        weights = torch.from_numpy(weights)
        
        return self.transform(img_z), self.transform(img_x), score_gt, weights

    def __len__(self):
        return self.itemCnt

if __name__ == '__main__':
    sets = VID("./datasets/imdb_video_train.json","./datasets/ILSVRC2015_VID_CURATION", phase="train")
    exem, srh, score_gt, weights = sets[8]
    print(exem.size(), srh.size(), score_gt.size(), weights.size())