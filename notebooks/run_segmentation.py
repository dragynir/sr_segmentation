
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
    ids_zero = np.where(data == 0)

    patch_data = patch.patchify(data, (8, 256, 256), 128)

    # data patches segmentation
    batch_size = 1

    if (patch_data.max() > 1):
        patch_data = patch_data / patch_data.max()

    N = patch_data.shape[0] * patch_data.shape[1] * patch_data.shape[2]
    patch_data_merged = patch_data.reshape((N, patch_data.shape[3], patch_data.shape[4], patch_data.shape[5]))

    vol = patch_data_merged
    vol = np.reshape(vol, vol.shape + (1,))
    result = model.predict(vol, verbose=1, batch_size=batch_size)
    segmented = result.reshape(patch_data.shape)

    print('recon 3d ...')
    recon = recon_3D(data_patches=segmented, patch_step=(8, 128, 128), patch_size=(16, 256, 256),
                     recon_shape=data.shape)

    recon = recon.astype(np.float32)

    # threshold
    recon[recon >= 0.8] = 1
    recon[recon < 0.8] = 0

    visualize(recon)


if __name__ == '__main__':

    # check availible GPUS and set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    # set mixed precision (32/16 bit) presision
    tf.keras.mixed_precision.experimental.set_policy('mixed_float16')

    batch_size = 1

    model = UNet_3D(input_size=(256, 256, 256, 1))
    model.load_weights("../models/UNet_3D_16_256.hdf5")

    img = cv2.imread('/home/d_korostelev/Projects/super_resolution/Real-ESRGAN/datasets/real/sub/sandstone/1x_1024/recon_00100_s011.png')
    img = img[:, :, 0]
    data = np.stack([img] * 16, axis=0)

    # np.ones((512, 512, 512))
    predict_volume(model, data)

    # debug
    # vol = np.ones((1, 256, 256, 256))
    # result = model.predict(vol, verbose=1, batch_size=batch_size)
    # print(result.shape)
