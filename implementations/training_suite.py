import os
import keras
import jsonpickle

from dataset import DatasetProvider, DatasetSchema
from variants.ConvVAEDescriptor import ConvLayer as cl, ConvVAEDescriptor
from variants.DenseVAEDescriptor import DenseVAEDescriptor
from variants.VAEDescriptor import VAEDescriptor

from PIL import Image

import numpy as np


class TrainingDataset:

    def __init__(self, name, dataset_schema: DatasetSchema) -> None:
        self.name = name
        self.dataset_scheme = dataset_schema

class TrainingConfiguration:

    def __init__(self, title: str, descriptor: VAEDescriptor) -> None:
        self.title = title
        self.descriptor = descriptor
        pass

class TrainingResult:

    def __init__(self, path, configuration, history, duration, batch_size, optimizer) -> None:
        self.path = path
        self.configuration = configuration
        self.history = history
        self.duration = duration
        self.batch_size = batch_size
        self.optimizer = optimizer
        pass

    def dump(self, filename):
        jsontext = jsonpickle.encode(self)
        path = os.path.join(self.path, filename)
        with open(path, 'wt') as file:
            file.write(jsontext)
        return


class Variations:

    def __init__(self, datasets: list[TrainingDataset], configurations: list[TrainingConfiguration], optimizers: list[str], batch_sizes: list[int]) -> None:
        self.datasets = datasets
        self.configurations = configurations
        self.optimizers = optimizers
        self.batch_sizes = batch_sizes
        pass
    
TRAINING_VARIATIONS : Variations = Variations(
    datasets = [
        # TrainingDataset("test", DatasetProvider.AvailableDatasets.TEST),
        # TrainingDataset("feret", DatasetProvider.AvailableDatasets.COLORFERET),
        TrainingDataset("ffhq", DatasetProvider.AvailableDatasets.FFHQ256),
    ],
    configurations = lambda img_shape: _generate_dense_configs(img_shape), #[ *_generate_conv_configs(img_shape), *_generate_dense_configs(img_shape)],
    optimizers = [ "adam" ], #adadelta schlecht, sgd und rmsprop haben nur nan
    batch_sizes = [ 16 ]
)

def _generate_dense_configs(img_shape):
    resize = [ 1/4, 1/2 ]
    latent_sizes = [16, 64] #128]
    layers = [
        [100],
        [500],
        [300, 100],
        [500, 100],
        [1000, 100],
        [1000, 300, 100],
        [3000, 1000, 100],
        [3000, 1000, 300, 100],
        # [3000, 2000, 500, 200, 100]
    ]
    # 9 * 3 * 2 = 54

    configs = []
    for f in resize:
        for s in latent_sizes:
            for l in layers:
                layer_name = "-".join(map(lambda ly: str(ly), l))
                configs.append(TrainingConfiguration("densevae_" + layer_name + "--" + str(s) + "--" + str(f) + "x", DenseVAEDescriptor(img_shape, l, s, f, "bilinear")))
    return configs

def _generate_conv_configs(img_shape):
    latent_sizes = [16, 64, 128]    
    prelatent_factor = [2, 5]
    layers: list[cl] = [
        [cl(3, 5, 2)],
        [cl(3, 5, 2), cl(5, 5)],
        [cl(3, 5, 2), cl(5, 5), cl(8, 3)],
        [cl(3, 5, 2), cl(5, 5), cl(8, 3), cl(10, 3)],
    ]
    # 3 * 2 * 5 = 25

    configs = []
    for latent in latent_sizes:
        for factor in prelatent_factor:
            for l in layers:
                layer_name = "-".join(map(lambda ly: str(ly.filters), l))
                configs.append(TrainingConfiguration("convvae_" + layer_name + "--" + str(latent) + "x" + str(factor), ConvVAEDescriptor(img_shape, l, factor, latent)))
    return configs

def saveImage(norm_img, filepath):
    gen_image = np.array(norm_img*255, dtype=np.uint8) # Load image from array and denormalize
    Image.fromarray(gen_image).save(filepath, format="png")



class ValidationSnapshotCallback(keras.callbacks.Callback):

    def __init__(self, val_images, directory):
        super().__init__()
        self.val_images = val_images
        self.directory = directory

    def on_epoch_end(self, epoch: int, logs=None):
        # Predict images
        gen_val_images = self.model(self.val_images)
        for i in range(len(gen_val_images)):
            filename = "e" + str(epoch) + "_" + str(i) + ".png"
            saveImage(gen_val_images[i], os.path.join(self.directory, filename))
