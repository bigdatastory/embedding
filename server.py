#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request

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
    print(content)
    content = content['detailParams']
    print('detailParams')
    print(content)
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
                                "description" : content
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

