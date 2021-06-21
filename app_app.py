#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## http://static.wooridle.net/lectures/chatbot/


# In[ ]:


#https://teamlab.github.io/jekyllDecent/blog/tutorials/%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%86%A1_%EC%B1%97%EB%B4%87_%EB%A7%8C%EB%93%A4%EA%B8%B0_with_python_flask_aws


# In[2]:


#https://korchris.github.io/2017/06/29/FB_chatbot/

from flask import Flask, request
import requests
app = Flask(__name__)
FB_API_URL = 'https://graph.facebook.com/v2.6/me/messages'
VERIFY_TOKEN='h_token_2021'
PAGE_ACCESS_TOKEN='EABUKa9a4QhkBABLFRnPDRZBUXS8AjsS6Rvnq4WOOp8G5dRJxk0MT4z1GNltZBktzhiIcCIH1P7dFdaGbB0tnZAhBWb4lXxEDpVpWbtCfoOO6U83czXFkUuxFWItg08KaFDSVRDPRFCo3jL0T0NngzepKZAfMvGwgF9QpN2qXJI6zZBs7upXNz'
                  
def send_message(recipient_id, text):
    """Send a response to Facebook"""
    payload = {
        'message': {
            'text': text
        },
        'recipient': {
            'id': recipient_id
        },
        'notification_type': 'regular'
    }

    auth = {
        'access_token': PAGE_ACCESS_TOKEN
    }

    response = requests.post(
        FB_API_URL,
        params=auth,
        json=payload
    )

    return response.json()

def get_bot_response(message):
    """This is just a dummy function, returning a variation of what
    the user said. Replace this function with one connected to chatbot."""
    return "This is a dummy response to '{}'".format(message)


def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = get_bot_response(message)
    send_message(sender, response)


def is_user_message(message):
    """Check if the message is a message from the user"""
    return (message.get('message') and
            message['message'].get('text') and
            not message['message'].get("is_echo"))


@app.route("/webhook", methods=['GET'])
def listen():
    """This is the main function flask uses to
    listen at the `/webhook` endpoint"""
    if request.method == 'GET':
        return verify_webhook(request)

@app.route("/webhook", methods=['POST'])
def talk():
    payload = request.get_json()
    event = payload['entry'][0]['messaging']
    for x in event:
        if is_user_message(x):
            text = x['message']['text']
            sender_id = x['sender']['id']
            respond(sender_id, text)

    return "ok"

@app.route('/')
def hello():
    return 'hello'

if __name__ == '__main__':
    app.run(threaded=True, port=5000)


# In[ ]:


##ttps://m.blog.naver.com/isaac7263/222082484001
##flask  웹서버 만들기


# In[ ]:





# In[ ]:




