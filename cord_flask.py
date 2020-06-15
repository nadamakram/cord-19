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
        json_ = request.json
        #dict input, str output
        strQuery = json.dumps(json_)
        #strQuery = json.loads(json_)
        strQuery = eval(strQuery)
        query = strQuery['question']
        
        documents_found = getMostSimilar(query)
        documents_found = str(documents_found)
        return documents_found

    except:

        return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    app.run(port=port, debug=True)