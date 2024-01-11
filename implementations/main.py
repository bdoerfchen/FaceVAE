import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ["DISPLAY"] = "localhost:0.0"

import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import keras

from basic.GadVAE import GadVAE
found_gpus = tf.config.list_physical_devices('GPU')

from dataset import DatasetProvider


MODEL_FILE = "vae-basic.keras"
BATCH_SIZE = 64

def main(load_model = False, save_model = True):
    dataset_meta = DatasetProvider.AvailableDatasets.COLORFERET
    dataset, path = DatasetProvider.loadDataset(dataset_meta, BATCH_SIZE)
    dataset = dataset.map(lambda x: (x/255, x/255))

    val_images = [ "00084/00084_931230_fb.ppm.png", "00114/00114_931230_fb.ppm.png", "00290/00290_940422_hl.ppm.png", "00551/00551_940519_fb.ppm.png", "00826/00826_940307_fa.ppm.png" ]
    #val_images = [ "00300/00300_940422_fa.ppm.png", "00300/00300_940422_hl.ppm.png", "00300/00300_940422_pl.ppm.png", "00400/00400_940422_hl.ppm.png" ]
    val_images = list(map( 
        lambda x: tf.convert_to_tensor(Image.open(os.path.join(path, x)), dtype=tf.float32)/255, 
        val_images))
    val_images = tf.stack(val_images)
    

    #vae = BasicVAE(dataset=dataset, image_shape=(384, 256, 3), latent_size=100)

    if not load_model:
        vae = GadVAE(img_shape=(384, 256, 3), latent_size=15)
        vae.fit(dataset, epochs=8, batch_size=BATCH_SIZE)
    else:
        vae = GadVAE.load_from_directory(".")
        
    if save_model:
        vae.save(".")


    im = np.array(val_images)
    test = np.array(vae.encoder(val_images))

    n_val_images = len(val_images)
    gen_images = vae.model.predict(val_images)
    f, axes = plt.subplots(2, n_val_images)
    for i in range(n_val_images):
        axes[0, i].imshow(np.array(val_images[i]*255, dtype=np.uint8))
        #val_image_in = np.reshape(val_images[i], newshape=(1, 384, 256, 3))
        #gen_image = np.reshape(vae.model(val_image_in), newshape=(384, 256, 3))
        gen_image = np.array(gen_images[i]*255, dtype=np.uint8)
        axes[1, i].imshow(gen_image)
    plt.show()
    e = 1



if __name__ == "__main__":
    main()