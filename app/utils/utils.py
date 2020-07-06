# from keras import backend as K

import io
import os
import re

import dill
import numpy as np
import PIL
import requests
import tensorflow as tf
import tensorflow_hub as hub
from keras.applications.mobilenet import MobileNet

from keras.applications.mobilenet import preprocess_input
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from PIL import Image

import wandb

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
os.environ["WANDB_API_KEY"] = "28aa171cff29182ca85739ae0a27685ad48c3d0d"


def preprocess_sentences(input_strings):
    """
    Preprocesses the sentence with a RegEx before being fed into the language model

    Args:
    input_strings (str) : input sentences - these are the sentences from the WhatsApp message

    Return:
        Processed sentences
    """
    return [
        re.sub(r"(covid-19|covid)", "coronavirus", input_string, flags=re.I)
        for input_string in input_strings
    ]


def get_lang_model():
    """
    Returns Universal Sentence Encoder (USE) from TF-HUB

    Return:
        USE language model
    """
    return hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
    )


def get_trained_lang_embedding():
    """
    Retrieves the trained language embedding from Weights and Biases for the WHO - FAQS

    Return:
        Returns a numpy array of the language embedding
    """
    dill_file = wandb.restore(
        "lang_embeddings.p", run_path="seedatnabeel/qa_lang_model//g1fo1le9"
    )
    # Loads the pickled file
    embed = dill.load(open(dill_file.name, "rb"))
    return embed


def get_model_weights():
    """
    Retrieves the trained CNN (Mask classifier) weights from Weights and Biases

    Return:
        Returns the path to the best CNN model weights
    """
    best_model = wandb.restore(
        "model-best.h5", run_path="seedatnabeel/mask_cv_model/34y9teh1"
    )
    return best_model


def build_cnn_model():
    """
    Builds the CNN Model for mask classification

    Return:
        Returns the built CNN model - without weights loaded
    """
    # load the base-model (Mobilenet), without imagenet weights and no classification head
    baseModel = MobileNet(weights=None, include_top=False, input_shape=IMAGE_SHAPE)

    # Add a classification head of fully connected layers with output 3 classes
    headModel = baseModel.output
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    # Return the model
    return Model(inputs=baseModel.input, outputs=headModel)


def bytes_to_numpy_array(content):
    """
    Converts the image bytes (from the WhatsApp request) to a numpy array

    Args:
    content (request) : Request  received from whatsapp

    Return:
        image_array - numpy array of the image
    """
    image_url = content.get("MediaUrl0")
    # Retreive the image url
    data = requests.get(image_url).content
    # Get the image bytes
    bytesobj = io.BytesIO(data)
    # Open the bytes obj as a PIL image
    img = Image.open(bytesobj)
    # Resize the image for the CNN input_shape
    img = img.resize((224, 224), Image.ANTIALIAS)
    # get the image as a numpy array
    image_array = (
        np.array(img.getdata())
        .astype(np.float32)
        .reshape((img.size[0], img.size[1], 3))
    )
    # expand dims so that it's a tensor
    image_array = np.expand_dims(image_array, axis=0)
    # Apply the imagenet pre-processing
    image_array = preprocess_input(image_array)
    return image_array
