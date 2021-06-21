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
    
    
    try:
        all_list = get_bot_response(content)
        
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
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

