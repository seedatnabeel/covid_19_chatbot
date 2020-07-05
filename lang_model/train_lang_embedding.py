import argparse
import logging
import os
import re

import dill
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import wandb

logging.getLogger().setLevel(logging.INFO)


def set_wandb_api_key(wandb_api):
    """
    Sets the wandb api key
    Args:

    wandb_api: api key for wandb
    """
    os.environ["WANDB_API_KEY"] = wandb_api


def load_use(url):
    """
    Loads Universal Sentence Encoder (USE) from url

    Args:

    url: url of USE

    Returns:
        lang_model: loaded language model
    """
    logging.info("Loading Universal Sentence Encoder")
    lang_model = hub.load(url)
    return lang_model


def load_ref_data(filepath):
    """
    Loads the reference FAQs from file to a pandas df

    Args:

    filepath: path to the reference FAQS

    Returns:
        data: pandas DataFrame
    """
    logging.info("Loading h5 of FAQ reference")
    data = pd.read_hdf(filepath, "df")
    return data


def preprocess_sentences(input_sentences):
    """ preprocessing of sentences for variants of covid"""
    return [
        re.sub(r"(covid-19|covid)", "coronavirus", input_sentence, flags=re.I)
        for input_sentence in input_sentences
    ]


def create_ref_embeddings(data, lang_model):
    """
    Creates a reference language embedding using USE

    Args:

    data: pandas DataFrame
    lang_model: loaded language embedding model

    Returns:
        ref_embedding: reference language embedding from the FAQS
    """
    logging.info("Creating Reference Embedding with USE")
    ref_embedding = lang_model.signatures["response_encoder"](
        input=tf.constant(preprocess_sentences(data.Answer)),
        context=tf.constant(preprocess_sentences(data.Question)),
    )["outputs"]
    return ref_embedding


def log_embedding_to_wandb(ref_embedding):
    """
    Logs lang embedding to wandb

    Args:

    ref_embedding: reference language embedding

    """

    logging.info("Logging the Embeddings to Weights and Biases")
    with open("lang_embeddings.p", "wb") as dill_file:
        dill.dump(ref_embedding, dill_file)

    run = wandb.init(project="qa_lang_model", job_type="producer")

    wandb.save("lang_embeddings.p")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="train lang model embeddings")
    parser.add_argument(
        "wandb_api", help="Wandb api key", type=str,
    )
    parser.add_argument(
        "--use_url",
        type=str,
        help="url for USE",
        default="https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3",
    )

    parser.add_argument(
        "--ref_filepath", type=str, help="Path to h5 ref FAQS", default="ref_faqs.h5",
    )

    args = parser.parse_args()

    wandb_api = args.wandb_api
    url = args.use_url

    filepath = args.ref_filepath

    set_wandb_api_key(wandb_api)
    lang_model = load_use(url)
    data = load_ref_data(filepath)
    ref_embedding = create_ref_embeddings(data, lang_model)
    log_embedding_to_wandb(ref_embedding)
