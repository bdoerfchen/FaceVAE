import os
import platform
from enum import Enum

import keras
import tensorflow

class DatasetSchema:
    def __init__(self, path, size, val_images) -> None:
        self.path = path
        self.size = size
        self.val_images = val_images
        pass

class DatasetProvider:    

    class AvailableDatasets:
        COLORFERET: DatasetSchema = DatasetSchema("feret", (384, 256, 3), [ "00084/00084_931230_fb.ppm.png", "00114/00114_931230_fb.ppm.png", "00290/00290_940422_hl.ppm.png", "00551/00551_940519_fb.ppm.png", "00826/00826_940307_fa.ppm.png" ])
        FFHQ256: DatasetSchema = DatasetSchema("ffhq256", (256, 256, 3), [ "00312.png", "41923.png", "12491.png", "41293.png", "34122.png", "54112.png", "25199.png", "50292.png"])
        TEST: DatasetSchema = DatasetSchema("test", (384, 256, 3), [ "00300/00300_940422_fa.ppm.png", "00300/00300_940422_hl.ppm.png", "00300/00300_940422_pl.ppm.png", "00400/00400_940422_hl.ppm.png" ])
        

    def getPath(dataset: AvailableDatasets):
        base_path = ""
        if platform.system() == "Linux":
            base_path = "./datasets/lin/"
        elif platform.system() == "Windows":
            base_path = ".\\datasets\\win\\"
        
        return os.path.join(os.curdir, base_path, dataset.path)

    def loadDataset(dataset: AvailableDatasets, batch_size = 64) -> tensorflow.data.Dataset:
        dir = DatasetProvider.getPath(dataset)
        size = list(dataset.size[0:2])
        assert os.path.exists(dir)
        dataset = keras.utils.image_dataset_from_directory(
            dir,
            labels=None, #No labels
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=size,
            shuffle=True,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        return dataset, dir
        
    
