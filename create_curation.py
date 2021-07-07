import os, sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import json
import pickle
import functools
import xml.etree.ElementTree as ET

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

from utils import get_instance_image

def worker(output_dir, video_dir):
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    image_names = sorted(image_names,
                        key=lambda x:int(x.split('/')[-1].split('.')[0]))
    video_name = video_dir.split('/')[-1]

    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    trajs = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        anno_name = image_name.replace('Data', 'Annotations').replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        fno = int(tree.find('filename').text)
        objs = tree.findall('object')
        for obj in objs:
            trackid = int(obj.find('trackid').text)
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            bbox = [x1, y1, x2, y2]
            if trackid not in trajs:
                trajs[trackid] = []
            trajs[trackid].append({'fno': fno, 'bbox': bbox})

            im_instance, _, _ = get_instance_image(img, bbox,
                    127, 255, 0.5)

            im_instance_name = os.path.join(save_folder, "{:06d}.{:02d}.x.jpg".format(fno, trackid))
            cv2.imwrite(im_instance_name, im_instance, [int(cv2.IMWRITE_JPEG_QUALITY), 90])            

    for trackid in list(trajs.keys()):
        if len(trajs[trackid]) < 2:
            del trajs[trackid]

    return {'name': video_name, 'trajs': trajs}

def processing(data_dir, output_dir, num_threads=32):
    # get all 4417 videos
    video_dir = os.path.join(data_dir, 'Data/VID')
    
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))

    videos = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Pool(processes=num_threads) as pool:
        for video in tqdm(pool.imap_unordered(
            functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            videos.append(video)

    validation_ratio = 0.0
    num_videos = len(videos)
    num_train_video = int(round(num_videos * (1 - validation_ratio)))
    num_val_video = num_videos - num_train_video

    imdb_video_train = {'num_videos': num_train_video,
                        'videos': videos[:num_train_video]}
    imdb_video_val = {'num_videos': num_val_video,
                      'videos': videos[num_train_video:]}

    # save imdb information
    json.dump(imdb_video_train, open(os.path.join(output_dir, '..', 'imdb_video_train.json'), 'w'), indent=2)
    json.dump(imdb_video_val, open(os.path.join(output_dir, '..', 'imdb_video_val.json'), 'w'), indent=2)


if __name__ == '__main__':
    Fire(processing)

