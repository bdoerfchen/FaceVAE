import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import keras

from BaseVAE import BaseVAE
from variants.ConvVAEDescriptor import ConvVAEDescriptor
from variants.DenseVAEDescriptor import DenseVAEDescriptor

found_gpus = tf.config.list_physical_devices('GPU')
print("Found GPU(s):", found_gpus)

from dataset import DatasetProvider

tf.config.optimizer.set_jit(True) #Enable XLA
BATCH_SIZE = 32

def main(load_model = False, save_model = False):

    dataset_meta = DatasetProvider.AvailableDatasets.TEST
    if not load_model:
        dataset, path = DatasetProvider.loadDataset(dataset_meta, BATCH_SIZE)
        dataset = dataset.map(lambda x: (x/255, x/255))
    else:
        path = DatasetProvider.getPath(dataset_meta)

    #val_images = [ "00084/00084_931230_fb.ppm.png", "00114/00114_931230_fb.ppm.png", "00290/00290_940422_hl.ppm.png", "00551/00551_940519_fb.ppm.png", "00826/00826_940307_fa.ppm.png" ]
    val_images = [ "00300/00300_940422_fa.ppm.png", "00300/00300_940422_hl.ppm.png", "00300/00300_940422_pl.ppm.png", "00400/00400_940422_hl.ppm.png" ]
    val_images = list(map( 
        lambda x: tf.convert_to_tensor(Image.open(os.path.join(path, x)), dtype=tf.float32)/255, 
        val_images))
    val_images = tf.stack(val_images)
    

    if not load_model:
        #vae = BaseVAE(ConvVAEDescriptor(img_shape=(384, 256, 3), latent_size=15), name="ConvVAE") #Conv
        vae = BaseVAE(DenseVAEDescriptor(img_shape=(384, 256, 3), encoder_layer_units=[10], latent_size=10))
        vae.summary()
        vae.compile(keras.optimizers.SGD())
        earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=15)
        history = vae.fit(dataset, epochs=1, batch_size=BATCH_SIZE, callbacks=[earlyStopping])
        print("Losses", vae.losses)
    else:
        vae = BaseVAE.load_from_directory(".")
        
    if save_model:
        vae.save(".")

    latent = vae.layers[1](val_images)
    sampling = vae.layers[2](latent)
    decoded = vae.layers[3](sampling)

    n_val_images = len(val_images)
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