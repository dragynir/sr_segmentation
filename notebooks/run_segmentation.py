
import sys
sys.path.insert(0,'../')

# from IO import*
from data import*
from Unet_models import*

import h5py as h5
import patchify as patch
import cv2


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

    vol = np.ones((1, 256, 256, 256))

    result = model.predict(vol, verbose=1, batch_size=batch_size)
    print(result.shape)
