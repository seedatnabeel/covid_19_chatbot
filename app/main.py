import logging

import numpy as np
import pandas as pd
import tensorflow_text
from fastapi import FastAPI
from starlette.requests import Request

from .utils.bot_response import *
from .utils.utils import *

logging.getLogger().setLevel(logging.INFO)

app = FastAPI()

logging.info("Building CNN model")
cnnmodel = build_cnn_model()

logging.info("Loading Mask classifier weights from Wandb")
weights_file = get_model_weights()
cnnmodel.load_weights(weights_file.name)

logging.info("Loading Reference language embedding from Wandb")
ref_embed = get_trained_lang_embedding()

logging.info("Loading Universal Sentence Encoder")
lang_model = get_lang_model()

logging.info("Loading h5 of FAQ reference")
pandas_df = pd.read_hdf('.ref_faqs.h5', 'df')


@app.get("/health")
def health_check():
    '''Healthcheck endpoint to see if API running fine'''
    return {"message": "API Online"}

@app.post("/bot")
async def bot(request: Request):
    """
    Receives a `WhatsApp post request`.
    Either a `text message` or an `image/photo` of a mask

    - Multi-language support: English, Afrikaans, French, German and a few more

     `RETURN`
     - Text message: Returns an appropriate contextutal response from the WHO COVID-19 FAQS
     - Image: classification of a mask as a cloth mask, n95 mask or surgical mask

    """

    content = await request.form()

    # Check if an image has been sent from the bot
    if content.get("NumMedia") != "0":
        x = bytes_to_numpy_array(content)
        images = np.vstack([x])
        # Predict class of mask
        classes = cnnmodel.predict(images, batch_size=1, verbose=0)[0]
        # Get the class idx
        type_mask = np.argmax(classes)
        # get the bot reply based on the class
        reply = mask_type_response(type_mask)
        return bot_twilio_response(reply)

    # Else get the Body of the msg - whatsapp text
    incoming_msg = content.get("Body")

    # Lower case the text
    incoming_msg = incoming_msg.strip().lower()

    # Check if the incoming msg is a greeting
    if incoming_msg in ("hello", "hi", "howsit", "hey", "hwsit"):
        reply = bot_greeting()
        return bot_twilio_response(reply)

    else:
        test_questions = [incoming_msg]

        # embed the question from WhatsApp with USE
        question_embed = lang_model.signatures["question_encoder"](
            tf.constant(preprocess_sentences(test_questions))
        )["outputs"]

        # Compute the inner produc of the question and reference embedding
        # Compute the nearest embedding and find the answer in the df
        test_responses = pandas_df.Answer[
            np.argmax(np.inner(question_embed, ref_embed), axis=1)
        ]

        reply = str(test_responses.values[0])

        return bot_twilio_response(reply)
