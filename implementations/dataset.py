import os
import platform
from enum import Enum

import keras

class DatasetProvider:

    class AvailableDatasets(Enum):
        COLORFERET = "feret"
        FFHQ = "ffhq256"

    def getPath(dataset: AvailableDatasets):
        base_path = ""
        if platform.system() == "Linux":
            base_path = "../datasets/linux/"
        elif platform.system() == "Windows":
            base_path = "..\\datasets\\win\\"
        
        return os.path.join(base_path, dataset)

    def loadDataset(dataset: AvailableDatasets):
        dir = DatasetProvider.getPath(dataset)
        dataset = keras.utils.image_dataset_from_directory(
            data_path,
            labels=None, #No labels
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=128,
            image_size=(256, 256),
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        
    
