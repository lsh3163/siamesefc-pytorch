# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
from PIL import Image, ImageDraw
from sklearn import metrics
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import transforms
from tracker import *
import cv2
import numpy as np
from model import *
from glob import glob
from utils import *
from collections import OrderedDict
from metrics import *
import matplotlib.pyplot as plt
from sklearn import metrics
parser = argparse.ArgumentParser(description='Object Tracking')
parser.add_argument('--trained_model', default='./weights/embed_clr_15.pth', type=str, help='pretrained model')
parser.add_argument('--test_folder', default='./otb13', type=str, help='folder path to input images')
parser.add_argument('--start_idx', default='0', type=int, help='temporal robustness evaluation')
parser.add_argument('--scale_ratio', default='1', type=float, help='spatial robustness evaluation')
parser.add_argument('--shift_x', default='1', type=float, help='shift x for spatial robustness evaluation')
parser.add_argument('--shift_y', default='1', type=float, help='shift y for spatial robustness evaluation')
parser.add_argument('--viz_graph', default='OPE', type=str, help='Graph Visualization Name')
args = parser.parse_args()

result_folder = './result/'

if not os.path.isdir(result_folder):
    os.mkdir(result_folder)


if __name__ == '__main__':
    # load net
    print('Loading weights from checkpoint (' + args.trained_model + ')')
    
    mean_suc = []
    
   
    video_list = sorted(os.listdir(args.test_folder))
    t = time.time()
    print(len(video_list))
    
    inference_time = 0
    start_idx = args.start_idx
    
    seq_iou = []
    num_of_frames = 0
    video_list.append("Jogging2")
#     video_list.append("Skating22")
#     video_list.append("Human42")
    for video in video_list:
        if "ipynb" in video:
            continue
        
        video_path = args.test_folder + "/" + video
        gt_path = video_path + "/groundtruth_rect.txt"
        
        if video=="Jogging":
            gt_path = video_path + "/groundtruth_rect.1.txt"
        elif video=="Jogging2":
            video_path = args.test_folder + "/Jogging"
            gt_path = video_path + "/groundtruth_rect.2.txt"
#         elif video=="Skating2":
#             gt_path = video_path + "/groundtruth_rect.1.txt"
#         elif video=="Skating22":
#             video_path = args.test_folder + "/Skating2"
#             gt_path = video_path + "/groundtruth_rect.2.txt"
#         elif video=="Human4":
#             gt_path = video_path + "/groundtruth_rect.1.txt"
#         elif video=="Human42":
# #             video_path = args.test_folder + "/Human4"
# #             gt_path = video_path + "/groundtruth_rect.2.txt"
#         elif video=="Human4" or video=="Panda":
#             continue
            
        video_path = video_path + "/img"
        f = open(gt_path)
        lines = f.readlines()
        images_list = sorted(glob(video_path + "/*.jpg"))
        
        if video=="David":
            images_list = images_list[300:770]
            
        tracker = SiamFCTracker(model_path=args.trained_model, gpu_id=0)
        
        video_iou = []
        
        init_bbox = []
        iou_lists = []
        
        num_of_frames += len(images_list)
        
        for idx, (line, img_path) in enumerate(zip(lines, images_list)):
            line = line.replace(" ", ",")
            line = line.replace("\t", ",")

            if idx < start_idx:
                continue
            
            # First Ground Truth
            if idx==start_idx:
                infer_st = time.time()
                first_frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                first_x, first_y, width, height = map(int, line.split(","))
                bbox = [int(first_x * args.shift_x), int(first_y * args.shift_y), int(width * args.scale_ratio), int(height * args.scale_ratio)]
                tracker.init(first_frame, bbox)
                iou_lists.append(1.0)
                inference_time += (time.time() - infer_st)
                                
            elif idx > start_idx:
                infer_st = time.time()
                cur_frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
                bbox = tracker.update(cur_frame)
                inference_time += (time.time() - infer_st)
                
                gt_x, gt_y, gt_width, gt_height = map(int, line.split(","))
                
                iou = get_iou([gt_x, gt_y, gt_width, gt_height], bbox)
                
                iou_lists.append(iou)
                video_iou.append(iou)
                cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_RGB2BGR)
                
                name = "./paper/" + video.lower() + ".txt"
               
                f_paper = open(name)
                f_lines = f_paper.readlines()
                f_line = f_lines[min(idx-1, len(f_lines)-1)]
                x, y, w, h = map(int, f_line.split(","))
                cur_frame = cv2.rectangle(cur_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                cur_frame = cv2.rectangle(cur_frame, (gt_x, gt_y), (gt_x+gt_width, gt_y+gt_height), (0, 0, 255), 2)
                cur_frame = cv2.rectangle(cur_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                os.makedirs('./result/' + video, exist_ok=True)
                cv2.imwrite("./result/" + video + "/" + str(idx) + ".jpg", cur_frame)
                
            
        
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,1]
        success_rate = []
        total_sum = 0
        success_sum = 0
        for threshold in thresholds:
            total_num = len(iou_lists)
            success_num = 0
            for iou in iou_lists:
                if iou > threshold:
                    success_num += 1    
                    success_sum += 1
                total_sum += 1
            success_rate.append(success_num / total_num)
        mean_suc.append(metrics.auc(thresholds, success_rate))
        seq_iou.append(success_rate)
        print(video, metrics.auc(thresholds, success_rate))
    
    seq_iou = np.array(seq_iou).mean(axis=0)
    
    mean_suc = np.array(mean_suc).mean(axis=0)
    print(mean_suc)
    print(metrics.auc(thresholds, seq_iou))
    plt.plot(thresholds, seq_iou ,label='Area = %0.3f)' % metrics.auc(thresholds, seq_iou))
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.autoscale(enable=True, axis='y', tight=True)
    plt.grid(color='#101010', alpha=0.5, ls=':')
    plt.legend(fontsize='medium')
    plt.xlabel('Thresholds')
    plt.ylabel('Success Rate')
    plt.title("One Pass Evaluation")
    plt.savefig(args.viz_graph + ".png")
    
    print("FPS : {} fps".format(num_of_frames / inference_time))
    print("elapsed time : {}s".format(time.time() - t))
