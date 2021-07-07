 
import numpy as np
    
def get_iou(gt, bbox):
    x1 = np.maximum(gt[0], bbox[0])
    y1 = np.maximum(gt[1], bbox[1])
    x2 = np.minimum(gt[0]+gt[2], bbox[2])
    y2 = np.minimum(gt[1]+gt[3], bbox[3])

    iou_width = np.maximum(x2 - x1 , 0)
    iou_height = np.maximum(y2 - y1, 0)
    iou_overlap = iou_width * iou_height

    areaA = gt[2] * gt[3]
    areaB = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return np.clip(iou_overlap / (areaA+areaB-iou_overlap), 0, 1)
