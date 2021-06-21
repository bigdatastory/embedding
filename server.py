#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    }
    return jsonify(dataSend)
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

