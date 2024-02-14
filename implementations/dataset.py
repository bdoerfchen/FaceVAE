import os
import platform
from enum import Enum

import keras
import tensorflow as tf

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
        FERETTEST: DatasetSchema = DatasetSchema("test", (384, 256, 3), [ "00300/00300_940422_fa.ppm.png", "00300/00300_940422_hl.ppm.png", "00300/00300_940422_pl.ppm.png", "00400/00400_940422_hl.ppm.png" ])
        FFHQTEST: DatasetSchema = DatasetSchema("ffhqtest", (256, 256, 3), [ "00122.png", "00127.png", "00132.png", "00137.png", "00142.png", "00147.png" ])
        

    def getPath(dataset: DatasetSchema):
        base_path = ""
        if platform.system() == "Linux":
            base_path = "./datasets/lin/"
        elif platform.system() == "Windows":
            base_path = ".\\datasets\\win\\"
        
        return os.path.join(os.curdir, base_path, dataset.path)

    def loadDataset(dataset: DatasetSchema, batch_size = 64, validation_split = 0.2, shuffle_seed = 1307) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """Loads a dataset by its DatasetScheme and splits it into shuffled training and validation subsets

        Args:
            dataset (DatasetSchema): the dataset to load
            batch_size (int): The batch size
            validation_split (float): The fraction of the validation split of the total dataset
            shuffle_seed (int): Seed for the pseudo-random shuffle
        Returns
            (dataset_training, dataset_validation, dataset_directory)
        """
        dir = DatasetProvider.getPath(dataset)  # Get path of dataset files
        size = list(dataset.size[0:2])          # Get (height, width)
        assert os.path.exists(dir)              # Make sure dataset exists
        dataset_train, dataset_validation = keras.utils.image_dataset_from_directory(
            dir,
            labels=None, #No labels
            label_mode=None,
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,
            image_size=size,
            shuffle=True,
            seed=shuffle_seed,
            validation_split=validation_split,
            subset="both",
            interpolation="bilinear",
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        return dataset_train, dataset_validation, dir
        
    
