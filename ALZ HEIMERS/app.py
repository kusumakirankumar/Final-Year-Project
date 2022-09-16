import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('kiran.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( final_features )

    if prediction==1:
        pred='Alzhiemer Disease Found!!'
    else:
        pred="Congratulations! You are Normal"
    output = pred

    return render_template('kiran.html', prediction_text='{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
