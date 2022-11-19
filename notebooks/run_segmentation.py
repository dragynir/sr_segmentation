
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0,'../')

# from IO import*
from data import*
from Unet_models import*

import h5py as h5
import patchify as patch
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd


def visualize(recon):
    print('visualize...')
    plt.figure(figsize=(25, 25))

    img = recon[250, :, :]
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')

    img = img * 255
    img = np.stack([img] * 3, axis=-1)
    cv2.imwrite('mask.png', img)
    plt.savefig('segmentation.png')


def predict_volume(model, data):

    patch_size = (256, 256, 256)
    patch_step = 128
    patch_data = patch.patchify(data, patch_size, patch_step)

    # data patches segmentation
    batch_size = 1

    if (patch_data.max() > 1):
        patch_data = patch_data / patch_data.max()

    N = patch_data.shape[0] * patch_data.shape[1] * patch_data.shape[2]
    patch_data_merged = patch_data.reshape((N, patch_data.shape[3], patch_data.shape[4], patch_data.shape[5]))

    vol = patch_data_merged
    vol = np.reshape(vol, vol.shape + (1,))
    result = model.predict(vol, verbose=1, batch_size=batch_size).squeeze()

    segmented = result.reshape(patch_data.shape)
    recon = patch.unpatchify(segmented, data.shape)  # no average mask (# TODO)

    # print('recon 3d ...')
    # recon = recon_3D(data_patches=segmented, patch_step=(patch_step, patch_step, patch_step), patch_size=patch_size,
    #                  recon_shape=data.shape)

    recon = recon.astype(np.float32)

    # threshold
    recon[recon >= 0.8] = 1
    recon[recon < 0.8] = 0

    return recon

    # visualize(recon)


def predict_images(model, df, source, out):
    paths = os.listdir(source)
    paths = list(filter(lambda x: '.png' in x, paths))
    paths = list(filter(lambda x: x in df.path.values, paths))

    print(df.material.value_counts())

    step = 256
    for i in tqdm(range(0, len(paths), step)):
        batch_paths = paths[i:i + step]
        volume = np.stack(tuple(cv2.imread(os.path.join(source, p))[:, :, 0] for p in batch_paths), axis=0)
        recon = predict_volume(model, volume)

        for p, ind in zip(batch_paths, range(recon.shape[0])):
            mask = recon[ind]
            mask = mask * 255
            mask = np.stack([mask] * 3, axis=-1)
            cv2.imwrite(os.path.join(out, f'mask_{p}'), mask)


if __name__ == '__main__':


    df = pd.read_csv('/home/d_korostelev/Projects/super_resolution/data/v1_dataset_DeepRockSR.csv')
    df = df[df.split == 'test']
    df = df[df.material == 'sandstone']

    # check availible GPUS and set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # set mixed precision (32/16 bit) presision
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    model = UNet_3D(input_size=(256, 256, 256, 1))
    model.load_weights("../models/UNet_3D_16_256.hdf5")

    source = '/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/tomo_test/'
    out = '../predictions/tomo_test'

    os.makedirs(out, exist_ok=True)

    predict_images(model, df, source, out)

    #
    # img = cv2.imread('/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sub/sandstone/1x_1024/recon_00100_s011.png')
    # img = img[:, :, 0]
    # data = np.stack([img] * 256, axis=0)
    # # np.ones((512, 512, 512))
    # predict_volume(model, data)

