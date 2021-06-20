#!/usr/bin/env python
# coding: utf-8

# In[6]:


from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

# 서버가 정상적으로 작동하는지 확인
@app.route("/")
def hello():
    return "Hello, Flask!"

@app.route('/coffe', methods=['POST'])
def coffe():
    req = request.get_json()
    
    coffe_menu = req["action"]["detailParams"]["coffe_menu"]["value"]
    
    answer = coffe_menu
    
    res = {
        "version" : "2.0",
        "template" : {
            "outputs" : [
                { "simpleText" : {
                        "text" : answer
                    }
                }
            ]
        }
    }
    
    return jsonify(res)

#메인함수
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True) # Flask 기본포트 5000번

