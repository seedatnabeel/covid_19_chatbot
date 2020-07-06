import argparse
import logging
import os

import tensorflow as tf
import wandb
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam
from wandb.keras import WandbCallback

from define_cnn import conv_net
from generator import data_generator

logging.getLogger().setLevel(logging.INFO)


wandb.init(project="mask_cv_model")


MODEL_PATH = "model_weights_final.h5"
WEIGHTS_PATH = "weights.hdf5"


def set_wandb_api_key(wandb_api):
    """
    Sets the wandb api key
    Args:

    wandb_api: api key for wandb
    """
    os.environ["WANDB_API_KEY"] = wandb_api


def train(data_dir, hyperparams):
    """Trains a pre-trained CNN by transfer learning and fine-tuning
       Log training metrics for evaluation and model weights to Weights and Biases
    """

    # Init the wandb project
    wandb.init(project="mask_cv_model")

    logging.info("Defining and Building the Conv Net")
    # Define and load the CNN -
    cnn = conv_net(
        pre_trained_model=hyperparams["pre_trained"],
        n_classes=hyperparams["n_classes"],
        dropout=hyperparams["dropout"],
        weights=hyperparams["weights"],
    )
    # Define your base pre-trained network
    baseModel = cnn.define_base_network()
    # Define the head network (fully connected layers for the task)
    cnn.define_head_network()

    # get the CNN model
    cnn_model = cnn.get_cnn()

    logging.info("Setting up data generators")
    # Set up data generators
    training_set, validation_set = data_generator(
        data_dir,
        batch_size=hyperparams["batch_size"],
        validation_size=hyperparams["validation_size"],
        horizontal_flip=hyperparams["horizontal_flip"],
        zoom_range=hyperparams["zoom_range"],
        shear_range=hyperparams["shear_range"],
        class_mode=hyperparams["class_mode"],
        target_size=(224, 224),
    )

    # Define early stopping criteria
    earlystop = EarlyStopping(
        monitor="val_loss", min_delta=0.0001, patience=3, verbose=1, mode="auto"
    )

    # model checkpointing
    checkpoint = ModelCheckpoint(
        WEIGHTS_PATH, monitor="val_loss", verbose=1, save_best_only=True, mode="max"
    )

    logging.info("Fitting dense layers - transfer learning")
    # Freeze the base model (pre-trained network) weights and only train the dense layers
    for layer in baseModel.layers:
        layer.trainable = False

    # compile the model
    cnn_model.compile(
        optimizer=Adam(lr=hyperparams["lr_dense"], decay=hyperparams["decay_dense"]),
        loss=hyperparams["loss"],
        metrics=["accuracy"],
    )

    # fit the generator and set callbacks
    cnn_model.fit_generator(
        training_set,
        steps_per_epoch=training_set.__len__(),
        epochs=hyperparams["epochs_dense"],
        validation_data=validation_set,
        validation_steps=validation_set.__len__(),
        use_multiprocessing=hyperparams["multiprocessing"],
        workers=8,
        callbacks=[earlystop, checkpoint, WandbCallback()],
    )

    logging.info("Fitting and fine-tuning the model")
    # Unfreeze the base Models to fine tune the network for the task
    for layer in baseModel.layers[15:]:
        layer.trainable = True

    # compile the model
    cnn_model.compile(
        optimizer=Adam(
            lr=hyperparams["lr_finetune"], decay=hyperparams["decay_finetune"]
        ),
        loss=hyperparams["loss"],
        metrics=["accuracy"],
    )

    # fit the generator and set callbacks
    cnn_model.fit_generator(
        training_set,
        epochs=hyperparams["epochs_finetune"] + hyperparams["epochs_dense"],
        steps_per_epoch=training_set.__len__(),
        validation_data=validation_set,
        validation_steps=validation_set.__len__(),
        initial_epoch=hyperparams["epochs_dense"],
        use_multiprocessing=hyperparams["multiprocessing"],
        workers=8,
        callbacks=[earlystop, checkpoint, WandbCallback()],
    )

    logging.info("Logging final and the best model to wandb")
    # Save and log the model to wandb
    cnn_model.save(os.path.join(wandb.run.dir, MODEL_PATH))
    K.clear_session()


def load_json(file_path):
    """Loads and returns a json file"""
    import json

    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train lang model embeddings")
    parser.add_argument(
        "wandb_api", help="Wandb api key", type=str,
    )

    parser.add_argument(
        "--data_dir", type=str, help="image data dir", default="my_data/masks",
    )

    parser.add_argument(
        "--hyperparams_path",
        type=str,
        help="Path to hyperparms file",
        default="hyperparams.json",
    )

    args = parser.parse_args()

    wandb_api = args.wandb_api
    data_dir = args.data_dir
    hyperparams_file = args.hyperparams_path

    set_wandb_api_key(wandb_api)
    hyperparams = load_json(hyperparams_file)
    train(data_dir, hyperparams)
