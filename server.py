#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, jsonify, request
from hp_rank_w2v import hp_rank_output, hp_rank_output_facebook 
from d_kind_model import d_kind_result
import requests

app = Flask(__name__)


###페이스북
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


#def get_bot_response(message):
#    """This is just a dummy function, returning a variation of what
#    the user said. Replace this function with one connected to chatbot."""
    
#    return hp_rank_w2v(message)

    
def verify_webhook(req):
    if req.args.get("hub.verify_token") == VERIFY_TOKEN:
        return req.args.get("hub.challenge")
    else:
        return "incorrect"

def respond(sender, message):
    """Formulate a response to the user and
    pass it on to a function that sends it."""
    response = hp_rank_output_facebook(message)
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



##카카오
@app.route('/keyboard')
def Keyboard():
    dataSend = {
      "user" : "has3ong",
      "blog" : "github",
    }
    return jsonify(dataSend)

@app.route('/message', methods=['POST'])
def Message():
    content = request.get_json()
    #content = content['userRequest']
    #content = content['utterance']
    content = content['action']
    content = content['params']
    content = content['user_symptom']
    
    
    try:
        all_list = hp_rank_output(content)
        
        hp_title = all_list[1]
        score_desc = all_list[2]
        imag_url =all_list[4]
        web_url = all_list[3]
        
        dataSend = {
          "version": "2.0",
          "template": {
            "outputs": [
              {
                "carousel": {
                  "type": "basicCard",
                  "items": [
                    {
                      "title":hp_title[0],
                      "description": score_desc[0],
                      "thumbnail": {
                        "imageUrl":imag_url[0]
                      },
                      "buttons": [
                        {
                          "action": "webLink",
                          "label": "더보기",
                          "webLinkUrl": "http://jic6405.dothome.co.kr/mobile.html?hp_nm=" + hp_title[0].split(':')[1]  
                        },
                        {
                          "action":  "webLink",
                          "label": "예약하기",
                          "webLinkUrl": web_url[0]
                        }
                      ]
                    },
                    {
                      "title": hp_title[1],
                      "description":score_desc[1],
                      "thumbnail": {
                        "imageUrl": imag_url[1]
                      },
                      "buttons": [
                        {
                          "action": "webLink",
                          "label": "더보기",
                          "webLinkUrl": "http://jic6405.dothome.co.kr/mobile.html?hp_nm=" + hp_title[1].split(':')[1]    
                        },
                        {
                          "action":  "webLink",
                          "label": "예약하기",
                          "webLinkUrl": web_url[1]
                        }
                      ]
                    },
                    {
                      "title": hp_title[2],
                      "description": score_desc[2],
                      "thumbnail": {
                        "imageUrl": imag_url[2]
                      },
                      "buttons": [
                        {
                          "action": "webLink",
                          "label": "더보기",
                          "webLinkUrl": "http://jic6405.dothome.co.kr/mobile.html?hp_nm=" + hp_title[2].split(':')[1]    
                        },
                        {
                          "action":  "webLink",
                          "label": "예약하기",
                          "webLinkUrl": web_url[2]
                        }
                      ]
                    },
                    {
                      "title": hp_title[3],
                      "description": score_desc[3],
                      "thumbnail": {
                        "imageUrl": imag_url[3]
                      },
                      "buttons": [
                        {
                          "action": "webLink",
                          "label": "더보기",
                          "webLinkUrl": "http://jic6405.dothome.co.kr/mobile.html?hp_nm=" + hp_title[3].split(':')[1]    
                        },
                        {
                          "action":  "webLink",
                          "label": "예약하기",
                          "webLinkUrl": web_url[3]
                        }
                      ]
                    },
                    {
                      "title": hp_title[4],
                      "description": score_desc[4],
                      "thumbnail": {
                        "imageUrl": imag_url[4]
                      },
                      "buttons": [
                        {
                          "action": "webLink",
                          "label": "더보기",
                          "webLinkUrl": "http://jic6405.dothome.co.kr/mobile.html?hp_nm=" + hp_title[4].split(':')[1]    
                        },
                        {
                          "action":  "webLink",
                          "label": "예약하기",
                          "webLinkUrl": web_url[4]
                        }
                      ]
                    },                  
                  ]
                }
              }
            ]
          }
        }
    except:
        dataSend = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel": {
                            "type" : "basicCard",
                            "items": [
                                {
                                    "title" : "",
                                    "description" : "검색결과를 찾을 수 없습니다.\n다시 검색해주세요.\n(Tip. 증상을 간략하게 작성하여 주세요.)"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    
    return jsonify(dataSend)


@app.route('/d_kind', methods=['POST'])
def Message():
    content = request.get_json()
    #content = content['userRequest']
    #content = content['utterance']
    content = content['action']
    content = content['params']
    content = content['user_symptom']
    
    
    try:
        d_kind_value = d_kind_result(content)
        
        disease_2 = disease_1[disease_1['질병명'] ==d_kind_value[0][0]]
        d_main_1 = disease_2[disease_2['항목'] == '증상']['항목내용'].values[0]
        
        disease_2 = disease_1[disease_1['질병명'] ==d_kind_value[1][0]]
        d_main_2 = disease_2[disease_2['항목'] == '증상']['항목내용'].values[0]
        
        disease_2 = disease_1[disease_1['질병명'] ==d_kind_value[2][0]]
        d_main_3 = disease_2[disease_2['항목'] == '증상']['항목내용'].values[0]
        
        disease_2 = disease_1[disease_1['질병명'] ==d_kind_value[3][0]]
        d_main_4 = disease_2[disease_2['항목'] == '증상']['항목내용'].values[0]
        
        disease_2 = disease_1[disease_1['질병명'] ==d_kind_value[4][0]]
        d_main_5 = disease_2[disease_2['항목'] == '증상']['항목내용'].values[0]
        
        {
          "version": "2.0",
          "template": {
            "outputs": [
              {
                "carousel": {
                  "type": "listCard",
                  "items": [
                    {
                      "header": {
                        "title": "질병 검색결과"
                      },
                      "items": [
                        {
                          "title": d_kind_value[0][0],
                          "description": d_main_1,
                        },
                        {
                          "title": d_kind_value[1][0],
                          "description": d_main_2,
                        },
                        {
                          "title": d_kind_value[2][0],
                          "description": d_main_3,
                        },
                        {
                          "title": d_kind_value[3][0],
                          "description": d_main_4,
                        },
                        {
                          "title": d_kind_value[4][0],
                          "description": d_main_5,
                        }
                      ]
                    }
                  ]
                }
              }
            ],
            "quickReplies": [
              {
                "messageText": "검색필터",
                "action": "message",
                "label": "검색필터"
              },
              {
                "messageText": "질병다시검색",
                "action": "message",
                "label": "질병다시검색"
              }
            ]
          }
        }
    
    
    except:
        dataSend = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "carousel": {
                            "type" : "basicCard",
                            "items": [
                                {
                                    "title" : "",
                                    "description" : "검색결과를 찾을 수 없습니다.\n다시 검색해주세요.\n(Tip. 증상을 간략하게 작성하여 주세요.)"
                                }
                            ]
                        }
                    }
                ]
            }
        }
    
    return jsonify(dataSend)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


# In[2]:


import numpy as np

np.__version__

