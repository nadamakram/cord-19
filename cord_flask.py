from flask import Flask, request, jsonify
import traceback
import pandas as pd
import numpy as np
import json
from nltk.tokenize import word_tokenize
from predict import getMostSimilar


app = Flask(__name__)

@app.route('/similar', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    try:
        #{"question": "what we know about risk factors?"}
        data = request.get_json()
        query = data['queryResult']['queryText']
        
        documents_found = getMostSimilar(query)
        documents_found = str(documents_found)
        #return jsonify({'Most similar articles':documents_found})
        return jsonify({ "fulfillmentMessages": [{"text": {"text": [documents_found]}}]})

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    
    port = 80 # If you don't provide any port then the port will be set to 12345
    app.run(port=port, debug=True,ssl_context='adhoc')


print(data)    
