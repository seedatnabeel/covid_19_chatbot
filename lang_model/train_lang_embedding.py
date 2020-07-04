import logging
import os
import re

import dill
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

import wandb

logging.getLogger().setLevel(logging.INFO)

os.environ["WANDB_API_KEY"] = "28aa171cff29182ca85739ae0a27685ad48c3d0d"

logging.info("Loading Universal Sentence Encoder")
lang_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual-qa/3"
)

logging.info("Loading reference FAQs")
data = pd.read_csv("WHO_FAQ.csv", encoding="utf8")  # datset link
data = data.rename(columns={"Context": "Question"})
# USE pretrained model to extract response encodings.


def preprocess_sentences(input_sentences):
    return [
        re.sub(r"(covid-19|covid)", "coronavirus", input_sentence, flags=re.I)
        for input_sentence in input_sentences
    ]


logging.info("Creating Reference Embedding with USE")
ref_embedding = lang_model.signatures["response_encoder"](
    input=tf.constant(preprocess_sentences(data.Answer)),
    context=tf.constant(preprocess_sentences(data.Question)),
)["outputs"]

with open("lang_embeddings.p", "wb") as dill_file:
    dill.dump(ref_embedding, dill_file)

logging.info("Logging the Embeddings to Weights and Biases")
run = wandb.init(project="qa_lang_model", job_type="producer")

wandb.save("lang_embeddings.p")
