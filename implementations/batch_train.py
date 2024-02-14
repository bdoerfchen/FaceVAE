import os
import sys
import time
import gc

import tensorflow as tf
from PIL import Image
import keras

from BaseVAE import BaseVAE
from dataset import DatasetProvider, DatasetSchema
from training_utilities import *

found_gpus = tf.config.list_physical_devices('GPU')
print("Found GPU(s):", found_gpus)
if len(found_gpus) == 0:
    raise Exception("GPU required for these computations")

tf.config.optimizer.set_jit(True) #Enable XLA

# Training constants
BATCH_SIZE = 16
MAX_EPOCHS = 100
EARLY_STOPPING_MINDELTA = 0.5

# Logging file constants
LOG_FILE = "stdout.log"
RESULT_FILE = "result.json"

# Version constants
MODEL_VERSION = "v1"
RESULT_PATH = "./models/" + MODEL_VERSION + "/"
SOFTSTOP_PATH = os.path.join(RESULT_PATH, "softstop.txt")

def setup() -> bool:
    """Prepares version folder and returns a bool indicating dir already existed"""
    new = not os.path.exists(RESULT_PATH)
    if new:
        os.makedirs(RESULT_PATH)
    return new

def main():
    """Main function that runs the training routines"""

    is_version_new = setup()
    if not is_version_new:
        print("Warning: Version already exists")
    
    # Train in all available datasets
    for dataset in TRAINING_VARIATIONS.datasets:

        # Load and normalize dataset
        loaded_dataset_train, loaded_dataset_val, _ = DatasetProvider.loadDataset(dataset.dataset_scheme)
        loaded_dataset_train = loaded_dataset_train.map(lambda x: (x/255, x/255))
        loaded_dataset_val = loaded_dataset_val.map(lambda x: (x/255, x/255))

        # Load a defined set of validation images to save for comparison
        val_images = load_val_images(dataset.dataset_scheme)
        subpath = os.path.join(RESULT_PATH, dataset.name)

        print("========\r\nLoaded dataset", dataset.name)

        # Load all configurations. This requires the img_size of the used dataset
        all_configurations = TRAINING_VARIATIONS.configurations(dataset.dataset_scheme.size)
        print("Loaded", len(all_configurations), "configurations")

        # Iterate over all possibilities for the optimizer, batch_size and training configurations
        for optimizer in TRAINING_VARIATIONS.optimizers:
            for batch_size in TRAINING_VARIATIONS.batch_sizes:
                for configuration in all_configurations:

                    # If specific file exists, stop training -- this is a soft-stop mechanism
                    if os.path.exists(SOFTSTOP_PATH):
                        print("Soft-stop detected. Stopping...")
                        return

                    time_start = time.perf_counter() # track when started to train
                    stdout_terminal = sys.stdout     # save stdout output stream for later
                    try:
                        title = optimizer + "_b" + str(batch_size) + "_" + configuration.title  # Determine title of specific configuration
                        variation_path = os.path.join(subpath, title)
                        os.makedirs(variation_path, exist_ok=True)                              # Create configuration dir and make sure all necessary dirs exists
                        logfile_path = os.path.join(variation_path, LOG_FILE)                   # Get the path to the new log file for this config
                        if os.path.exists(logfile_path):                                        # If configuration exists, skip
                            print("\tSkip ", title)
                            continue

                        print("\tStart ", title)
                        with open(logfile_path, "xt") as logfile:
                            sys.stdout = logfile                                                # Set logfile as temporary target of stdout to capture model summary, etc.
                            print("====== Training model (" + title + ") ======")

                            # Define and compile model from configuration data
                            keras.backend.clear_session()
                            model : BaseVAE = BaseVAE(descriptor=configuration.descriptor)      # Load VAE with descriptor of configuration
                            model.summary()
                            model.compile(optimizer=optimizer)                                  # Compile with optimizer of this run

                            # Create callbacks and fit model
                            sys.stdout = stdout_terminal                                        # Reset stdout, as file is too slow to also capture keras output..
                            earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=EARLY_STOPPING_MINDELTA, restore_best_weights=True, mode="min", start_from_epoch=5)
                            snapshotCallback = ValidationSnapshotCallback(val_images=val_images, directory=variation_path)
                            history = model.fit(loaded_dataset_train, 
                                                epochs=MAX_EPOCHS, 
                                                batch_size=batch_size, 
                                                callbacks=[earlyStopping, snapshotCallback], 
                                                validation_data=loaded_dataset_train, validation_batch_size=batch_size, 
                                                shuffle=True)

                            # Save model and results
                            sys.stdout = logfile                                                # Switch stdout back to file, to log next output
                            duration = time.perf_counter() - time_start                         # Calculate run duration
                            print("\r\nFinished after", duration, "s")
                            model.save_to_directory(variation_path)                             # Save model
                            history.model = None                                                # Remove model from history object to save space

                            # Combine all information into result object and dump as json in result file
                            result = TrainingResult(variation_path, configuration=configuration, history=history, duration=duration, batch_size=batch_size, optimizer=optimizer)
                            result.dump(RESULT_FILE)

                    except Exception as error:
                        # Log errors occured while training one config
                        print("Exception while running")
                        print("\t", error)
                    finally:
                        # After config, restore stdout
                        sys.stdout = stdout_terminal
                        print("\t\tFinished training")

        # To prevent memory leaking when loading the next dataset, delete the current one and start garbage collector
        del loaded_dataset_train
        del loaded_dataset_val
        gc.collect()
    
    pass


if __name__ == "__main__":
    main()

