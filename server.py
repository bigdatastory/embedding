#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request
from hp_rank_w2v import hp_rank_output as get_bot_response

app = Flask(__name__)

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
    
    """
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
                                "description" : get_bot_response(content)
                            }
                        ]
                    }
                }
            ]
        }
    }"""
    
    dataSend = {
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "carousel": {
              "type": "basicCard",
              "items": [
                {
                  "title":get_bot_response(content)[1][0],
                  "description": get_bot_response(content)[2][0],
                  "thumbnail": {
                    "imageUrl":get_bot_response(content)[4][0]
                  },
                  "buttons": [
                    {
                      "action": "message",
                      "label": "더보기",
                      "messageText": "분석결과는 다음과 같습니다.(개발중)"
                    },
                    {
                      "action":  "webLink",
                      "label": "예약하기",
                      "webLinkUrl": get_bot_response(content)[3][0]
                    }
                  ]
                },
                {
                  "title": get_bot_response(content)[1][1],
                  "description":get_bot_response(content)[2][1],
                  "thumbnail": {
                    "imageUrl": get_bot_response(content)[4][1]
                  },
                  "buttons": [
                    {
                      "action": "message",
                      "label": "더보기",
                      "messageText": "분석결과는 다음과 같습니다.(개발중)"
                    },
                    {
                      "action":  "webLink",
                      "label": "예약하기",
                      "webLinkUrl": get_bot_response(content)[3][1]
                    }
                  ]
                },
                {
                  "title": get_bot_response(content)[1][2],
                  "description": get_bot_response(content)[2][2],
                  "thumbnail": {
                    "imageUrl": get_bot_response(content)[4][2]
                  },
                  "buttons": [
                    {
                      "action": "message",
                      "label": "더보기",
                      "messageText": "분석결과는 다음과 같습니다.(개발중)"
                    },
                    {
                      "action":  "webLink",
                      "label": "예약하기",
                      "webLinkUrl": get_bot_response(content)[3][2]
                    }
                  ]
                },
                {
                  "title": get_bot_response(content)[1][3],
                  "description": get_bot_response(content)[2][3],
                  "thumbnail": {
                    "imageUrl": get_bot_response(content)[4][3]
                  },
                  "buttons": [
                    {
                      "action": "message",
                      "label": "더보기",
                      "messageText": "분석결과는 다음과 같습니다.(개발중)"
                    },
                    {
                      "action":  "webLink",
                      "label": "예약하기",
                      "webLinkUrl": get_bot_response(content)[3][3]
                    }
                  ]
                },
                {
                  "title": get_bot_response(content)[1][4],
                  "description": get_bot_response(content)[2][4],
                  "thumbnail": {
                    "imageUrl": get_bot_response(content)[4][4]
                  },
                  "buttons": [
                    {
                      "action": "message",
                      "label": "더보기",
                      "messageText": "분석결과는 다음과 같습니다.(개발중)"
                    },
                    {
                      "action":  "webLink",
                      "label": "예약하기",
                      "webLinkUrl": get_bot_response(content)[3][4]
                    }
                  ]
                },                  
              ]
            }
          }
        ]
      }
    }
    
    return jsonify(dataSend)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

