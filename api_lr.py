from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

IMAGES_FOLDER = os.path.join('static', 'images')

# API definition
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

@app.route('/')
def home():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'heart.jpg')
    return render_template('home.html', image = filename)

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST']) #use decorator pattern for the route
def predict():
    if lr:
        try:
            features = [float(x) for x in request.form.values()]
            final_features = [np.array(features)]
            #scaler = StandardScaler()
            #final_features = scaler.transform(final_features)    
            prediction = lr.predict(final_features)
            print("final features",final_features)
            print("prediction:",prediction)
            output = round(prediction[0], 2)
            print(output)

            if output == 0:
                return render_template('index.html', prediction_text='THE PATIENT IS NOT LIKELY TO HAVE A HEART DISEASE')
            else:
                return render_template('index.html', prediction_text='THE PATIENT IS LIKELY TO HAVE A HEART DISEASE')

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 8085
        
    # Load "model.pkl"
    lr = joblib.load(r'C:/Users/User/Desktop/Hear_beat/heart_disease/data/model_lrm.pkl')
    print ('Model loaded')
    # Load "model_columns.pkl"
    model_columns = joblib.load(r'C:/Users/User/Desktop/Hear_beat/heart_disease/data/model_columns_lrm.pkl')
    print ('Model columns loaded')
    app.run(port=port, debug=True)
