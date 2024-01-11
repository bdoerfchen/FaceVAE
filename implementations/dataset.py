import os
import platform
from enum import Enum

import keras
import tensorflow

class DatasetSchema:
    def __init__(self, path, size) -> None:
        self.path = path
        self.size = size
        pass

class DatasetProvider:    

    class AvailableDatasets:
        COLORFERET = DatasetSchema("feret", (384, 256))
        FFHQ256 = DatasetSchema("ffhq256", (256, 256))
        TEST = DatasetSchema("test", (384, 256))
        

    def getPath(dataset: AvailableDatasets):
        base_path = ""
        if platform.system() == "Linux":
            base_path = "./datasets/lin/"
        elif platform.system() == "Windows":
            base_path = ".\\datasets\\win\\"
        
        return os.path.join(os.curdir, base_path, dataset.path)

    def loadDataset(dataset: AvailableDatasets, batch_size = 64) -> tensorflow.data.Dataset:
        dir = DatasetProvider.getPath(dataset)
        assert os.path.exists(dir)
        dataset = keras.utils.image_dataset_from_directory(
            dir,
            labels=None, #No labels
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=dataset.size,
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        return dataset, dir
        
    
