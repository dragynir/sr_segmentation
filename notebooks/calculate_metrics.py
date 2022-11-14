import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib as plt


def read_mask(path):
    mask = cv2.imread(path)
    return mask[:, :, 0] / 255


def iou(gt, pred):
    inter = gt * pred
    union = gt.sum() + pred.sum() - inter
    return inter / (union + 1e-16)


def plot_iou(metrics):
    plt.figure()
    plt.hist(metrics, bins=100)
    plt.savefig('iou.png')


if __name__ == '__main__':

    target = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sub/sandstone/1x_1024/'
    pred = '../predictions/1x_1024'

    metrics = []
    paths = os.listdir(target)
    for name in tqdm(paths):
        gt_mask = read_mask(os.path.join(target, name))
        pred_mask = read_mask(os.path.join(pred, name))
        metrics.append(iou(gt_mask, pred_mask))

    avg = np.mean(metrics)

    print('Mean avg:', avg)
