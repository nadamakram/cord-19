from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
import json
from nltk.tokenize import word_tokenize

app = Flask(__name__)


@app.route('/similar', methods=['POST']) # Your API endpoint URL would consist /predict
def predict():
    if model:
        try:
            #{"question": "what we know about risk factors?"}
            json_ = request.json
            #dict input, str output
            strQuery = json.dumps(json_)
            #strQuery = json.loads(json_)
            strQuery = eval(strQuery)
            query = strQuery['question']
            queryVector = model.infer_vector(word_tokenize(query))
            similar_doc = model.docvecs.most_similar([queryVector],topn=20)
            similar_doc = str(similar_doc)
            #return jsonify({'Most similar articles': similar_doc})
            #return similar_doc
            return similar_doc

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line argument
    except:
        port = 12345 # If you don't provide any port then the port will be set to 12345
    model = joblib.load('nlp_model.pkl') # Load "model.pkl"
    print ('Model loaded')
    app.run(port=port, debug=True)        