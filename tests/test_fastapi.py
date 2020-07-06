from starlette.testclient import TestClient

from app import main

client = TestClient(main)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"message": "API Online"}

def test_bot_hello_endpoint():

    resp = client.post("/bot",data=
        {'SmsMessageSid':'SM612470e4c57f7e7edf1951cd09c1138e',
        'NumMedia': '0',
        'SmsSid': 'SM612470e4c57f7e7edf1951cd09c1138e',
        'SmsStatus': 'received',
        'Body': 'Hello',
        'To': 'whatsapp:+14155238886',
        'NumSegments': '1',
        'MessageSid': 'SM612470e4c57f7e7edf1951cd09c1138e',
        'AccountSid': 'AC1102d7cc2c23d58e0fdf5c6d5544cb80',
        'From': 'whatsapp:+27726961505',
         'ApiVersion': '2010-04-01'})

    assert resp.status_code == 200


def test_bot_QA_endpoint():

    resp = client.post("/bot",data=
        {'SmsMessageSid':'SM612470e4c57f7e7edf1951cd09c1138e',
        'NumMedia': '0',
        'SmsSid': 'SM612470e4c57f7e7edf1951cd09c1138e',
        'SmsStatus': 'received',
        'Body': 'What is Covid?',
        'To': 'whatsapp:+14155238886',
        'NumSegments': '1',
        'MessageSid': 'SM612470e4c57f7e7edf1951cd09c1138e',
        'AccountSid': 'AC1102d7cc2c23d58e0fdf5c6d5544cb80',
        'From': 'whatsapp:+27726961505',
         'ApiVersion': '2010-04-01'})

    assert resp.status_code == 200


def test_bot_image_endpoint():

    resp = client.post("/bot",data=
        {'MediaContentType0':'image/jpeg',
       'SmsMessageSid':'MMf17ae24039efd4e19fbc4c1b42fda9f8',
       'NumMedia':'1',
       'SmsSid':'MMf17ae24039efd4e19fbc4c1b42fda9f8',
       'SmsStatus':'received',
       'Body': '',
       'To':'whatsapp:+14155238886',
       'NumSegments':'1',
       'MessageSid':'MMf17ae24039efd4e19fbc4c1b42fda9f8',
       'AccountSid': 'AC1102d7cc2c23d58e0fdf5c6d5544cb80',
       'From': 'whatsapp:+27726961505',
       'MediaUrl0':'https://api.twilio.com/2010-04-01/Accounts/AC1102d7cc2c23d58e0fdf5c6d5544cb80/Messages/MMf17ae24039efd4e19fbc4c1b42fda9f8/Media/ME79963175e7bbff089a0805b0aeb96b1d'})

    assert resp.status_code == 200
