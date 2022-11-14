
import sys
sys.path.insert(0,'../')

# from IO import*
from data import*
from Unet_models import*

import h5py as h5
import patchify as patch
import cv2


def visualize(recon):
    plt.figure(figsize=(25, 25))

    plt.subplot(1, 3, 1)
    plt.imshow(recon[256, :, :], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(recon[:, 612, :], cmap='gray')

    plt.subplot(1, 3, 3)
    plt.imshow(recon[:, :, 612], cmap='gray')

    plt.savefig('segmentation.png')


def predict_volume(model, data):
    ids_zero = np.where(data == 0)

    patch_data = patch.patchify(data, (256, 256, 256), 128)

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

    recon = recon_3D(data_patches=segmented, patch_step=(128, 128, 128), patch_size=(256, 256, 256),
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

    predict_volume(model, np.ones((512, 512, 512)))

    # debug
    # vol = np.ones((1, 256, 256, 256))
    # result = model.predict(vol, verbose=1, batch_size=batch_size)
    # print(result.shape)
