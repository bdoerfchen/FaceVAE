import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import keras

from BaseVAE import BaseVAE
from dataset import DatasetProvider, DatasetSchema

from training_utilities import *

found_gpus = tf.config.list_physical_devices('GPU')
print("Found GPU(s):", found_gpus)


tf.config.optimizer.set_jit(True) #Enable XLA
BATCH_SIZE = 32

model_dir = "/mnt/d/Downloads/deepl-auswertung/models/wir/ffhq/adam_b16_densevae_1000-300--64--0.25x"

def main():

    dataset_meta: DatasetSchema = DatasetProvider.AvailableDatasets.FFHQTEST
    
    val_images = load_val_images(dataset_meta)
    n_val_images = len(val_images)
    
    vae = BaseVAE.load_from_directory(model_dir)

    gen_images = vae.predict(val_images)
    f, axes = plt.subplots(2, n_val_images, num="VAE validation")
    for i in range(n_val_images):
        _show_image_on_axis(axes[0, i], np.array(val_images[i]*255, dtype=np.uint8))
        gen_image = np.array(gen_images[i]*255, dtype=np.uint8) # Load image from array and denormalize
        _show_image_on_axis(axes[1, i], gen_image)
    axes[0, 0].set_title("Reference", loc="left")
    axes[1, 0].set_title("Reconstructed", loc="left")

    plt.show()
    e = 1

def _show_image_on_axis(faxis, image) -> None:
    faxis.imshow(image)
    faxis.get_xaxis().set_visible(False)
    faxis.get_yaxis().set_visible(False)
    pass


if __name__ == "__main__":
    main()