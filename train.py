import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim


from math import exp
from datasets import *
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import transforms
from model import *
from torch.autograd import Variable
from multiprocessing import Pool

#3.2768e-5
random.seed(42)

parser = argparse.ArgumentParser(description='Siamese FC Reimplementation')
parser.add_argument('--epochs', default=100, type = int,
                    help='epochs of training')
parser.add_argument('--batch_size', default=32, type = int,
                    help='batch size of training')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.8685, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--step_size', default=1, type=float,
                    help='Step size for SGD')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')


args = parser.parse_args()


if __name__ == '__main__':

    
    
    print("[*] Load Model")
    net = SiamFC() 
    torch.multiprocessing.set_start_method('spawn')
    
    net = net.cuda()
    net = torch.nn.DataParallel(net,device_ids=[0, 1, 2, 3, 4 ,5 ,6, 7]).cuda()
    
    cudnn.benchmark = True
    net.train()
    print("[*] Load Dataset")
            
    train_dataset = VID("./datasets/imdb_video_train.json","./datasets/ILSVRC2015_VID_CURATION", "train")
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=args.step_size, 
                gamma=args.gamma)
    step_index = 0
    
    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(args.epochs):
        train_time_st = time.time()
        loss_value = 0

        st = time.time()
     
        for index, (exem_img, srh_img, score_gt, weights) in enumerate(tqdm(data_loader)):

            exem_img = Variable(exem_img.type(torch.FloatTensor)).cuda()
            srh_img = Variable(srh_img.type(torch.FloatTensor)).cuda()
            score_gt = Variable(score_gt.type(torch.FloatTensor)).cuda()
            weights = Variable(weights.type(torch.FloatTensor)).cuda()

            optimizer.zero_grad()
    
            score_out = net(exem_img, srh_img)
                        
            
            a = -(score_gt * score_out)
            b = F.relu(a)
            logistic_loss = b + torch.log(torch.exp(-b) + torch.exp(a-b))
            logistic_loss = weights * logistic_loss
            loss = torch.sum(logistic_loss) / args.batch_size
            
            loss.backward()
            optimizer.step()
            
            loss_value += loss.item()
            
        scheduler.step()
        et = time.time()
        print(loss_value)
        print('Saving state, iter:', epoch)
        torch.save(net.state_dict(),
                   './weights/embed_clr_' + repr(epoch) + '.pth')