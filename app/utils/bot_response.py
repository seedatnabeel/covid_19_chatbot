from fastapi import Response
from twilio.twiml.messaging_response import MessagingResponse


def bot_twilio_response(reply):
    """
    Formats the bot reply message as a Twilio responses

    Args:
    reply (str) : Twilio message reply string

    Return:
        Twilio response message
    """
    response = MessagingResponse()
    response.message(str(reply))
    return Response(content=str(response), media_type="application/xml")


def mask_type_response(type_mask):
    """
    Returns a message based on the classification of the type of mask

    Args:
    type_mask (int) : Integer of the type of mask
                      0 = Cloth, 1 = N95, 2 = Surgical

    Return:
        Response message of the type of mask the CNN classified
    """
    # Cloth mask
    if type_mask == 0:
        reply = "You have a cloth mask - a surgical or n95 would be better"
    # N95 Mask
    elif type_mask == 1:
        reply = "You have an N95 Mask --- Yay that's the most protective mask"
    # Surgical mask
    elif type_mask == 2:
        reply = "You have a Surgical mask --- you are well protected"
    return reply


def bot_greeting():
    """
    Returns the bot greeting welcome string

    Return:
        Welcome/Greeting string
    """
    reply = (
        "Hello and welcome to your personal Covid-19 Chatbot!\n\n"
        "You can ask any Q's you have about Covid-19 - we support English, Afrikaans, French, German & more (just try and see)\n"
        "*** You can also take a picture of your mask and we'll tell you what type it is:\n\n"
        " Example questions\n"
        "- What is Covid-19\n"
        "- How do I get Covid-19\n"
        "- Just try different questions or a different language\n"
    )
    return reply
