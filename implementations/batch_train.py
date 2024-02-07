import os
import sys
import time
import gc

import tensorflow as tf
from PIL import Image
import keras

from BaseVAE import BaseVAE
from dataset import DatasetProvider, DatasetSchema
from training_suite import *

found_gpus = tf.config.list_physical_devices('GPU')
print("Found GPU(s):", found_gpus)
if len(found_gpus) == 0:
    raise Exception("GPU required for these computations")

tf.config.optimizer.set_jit(True) #Enable XLA
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_MINDELTA = 0.5

LOG_FILE = "stdout.log"
RESULT_FILE = "result.json"


MODEL_VERSION = "finaldense"
RESULT_PATH = "./models/" + MODEL_VERSION + "/"
SOFTSTOP_PATH = os.path.join(RESULT_PATH, "softstop.txt")
def setup() -> bool:
    new = not os.path.exists(RESULT_PATH)
    if new:
        os.makedirs(RESULT_PATH)
    return new

def load_val_images(dataset: DatasetSchema):
    path = DatasetProvider.getPath(dataset)
    val_images = dataset.val_images
    val_images = list(map( 
        lambda x: tf.convert_to_tensor(Image.open(os.path.join(path, x)), dtype=tf.float32)/255, 
        val_images))
    return tf.stack(val_images)

def main():
    new = setup()
    if not new:
        print("Warning: Version already exists")
    
    for dataset in TRAINING_VARIATIONS.datasets:

        # Load and normalize dataset
        loaded_dataset, _ = DatasetProvider.loadDataset(dataset.dataset_scheme)
        loaded_dataset = loaded_dataset.map(lambda x: (x/255, x/255))
        val_images = load_val_images(dataset.dataset_scheme)
        subpath = os.path.join(RESULT_PATH, dataset.name)
        print("========\r\nLoaded dataset", dataset.name)

        # Iterate over all possibilities
        all_configurations = TRAINING_VARIATIONS.configurations(dataset.dataset_scheme.size)
        print("Loaded", len(all_configurations), "configurations")
        for optimizer in TRAINING_VARIATIONS.optimizers:
            for batch_size in TRAINING_VARIATIONS.batch_sizes:
                for configuration in all_configurations: # Get configurations dependent on img_shape
                    if os.path.exists(SOFTSTOP_PATH):
                        print("Soft-stop detected. Stopping...")
                        return

                    time_start = time.perf_counter()
                    stdout_terminal = sys.stdout
                    try:
                        title = optimizer + "_b" + str(batch_size) + "_" + configuration.title
                        variation_path = os.path.join(subpath, title)
                        os.makedirs(variation_path, exist_ok=True) #Make sure dir exists
                        logfile_path = os.path.join(variation_path, LOG_FILE)
                        if os.path.exists(logfile_path):
                            print("\tSkip ", title)
                            continue # Skip this configuration, because it already exists

                        print("\tStart ", title)
                        with open(logfile_path, "xt") as logfile:
                            sys.stdout = logfile
                            print("====== Training model (" + title + ") ======")

                            # Define and compile model from configuration data
                            keras.backend.clear_session()
                            model : BaseVAE = BaseVAE(descriptor=configuration.descriptor)
                            model.summary()
                            model.compile(optimizer=optimizer)

                            # Create callbacks and fit model
                            sys.stdout = stdout_terminal
                            earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, min_delta=EARLY_STOPPING_MINDELTA, restore_best_weights=True, mode="min", start_from_epoch=5)
                            snapshotCallback = ValidationSnapshotCallback(val_images=val_images, directory=variation_path)
                            history = model.fit(loaded_dataset, epochs=MAX_EPOCHS, batch_size=batch_size, callbacks=[earlyStopping, snapshotCallback])
                            history.model = None    # Remove model from history object to save space
                            sys.stdout = logfile

                            # Save model and results
                            duration = time.perf_counter() - time_start
                            print("\r\nFinished after", duration, "s")
                            model.save_to_directory(variation_path)
                            model.count_params()
                            result = TrainingResult(variation_path, configuration=configuration, history=history, duration=duration, batch_size=batch_size, optimizer=optimizer)
                            result.dump(RESULT_FILE)

                            # Validate model saving
                            valVae = BaseVAE.load_from_directory(variation_path)
                            gen_val_images = valVae(val_images)
                            for i in range(len(gen_val_images)):
                                gen_image = np.array(gen_val_images[i]*255, dtype=np.uint8) # Load image from array and denormalize
                                filename = "val_" + str(i) + ".png"
                                Image.fromarray(gen_image).save(os.path.join(variation_path, filename), format="png")

                    except Exception as error:
                        print("Exception while running")
                        print("\t", error)
                    finally:
                        sys.stdout = stdout_terminal
                        print("\t\tFinished training")

        del loaded_dataset
        gc.collect()
    
    pass


if __name__ == "__main__":
    main()

