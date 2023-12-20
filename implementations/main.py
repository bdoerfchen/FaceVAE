import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import keras
found_gpus = tf.config.list_physical_devices('GPU')


from basic.BasicVAE import BasicVAE
from dataset import DatasetProvider


MODEL_FILE = "vae-basic.keras"
BATCH_SIZE = 64

def main(load_model = False, save_model = True):
    dataset, path = DatasetProvider.loadDataset(DatasetProvider.AvailableDatasets.COLORFERET, BATCH_SIZE)
    val_images = [ "00084/00084_931230_fb.ppm.png", "00114/00114_931230_fb.ppm.png", "00290/00290_940422_hl.ppm.png", "00551/00551_940519_fb.ppm.png", "00826/00826_940307_fa.ppm.png" ]
    val_images = list(map( 
        lambda x: tf.convert_to_tensor(Image.open(os.path.join(path, x)), dtype=tf.float32), 
        val_images))

    vae = BasicVAE(dataset=dataset, image_shape=(384, 256, 3), latent_size=100)
    if not load_model:
        vae.fit(epochs=2, batch_size=BATCH_SIZE)
    else:
        vae.vae.load_weights(MODEL_FILE)
        
    if save_model:
        vae.vae.save(MODEL_FILE, save_format="keras")


    n_val_images = len(val_images)
    f, axes = plt.subplots(2, n_val_images)
    for i in range(n_val_images):
        val_image = val_images[i]
        axes[0, i].imshow(val_image)
        axes[1, i].imshow(vae.reconstruct(val_image))
    f.show()



if __name__ == "__main__":
    main()