from flask import Flask, jsonify,request, url_for, redirect, render_template
import pickle
import numpy as np
import requests
import pandas as pd
app = Flask(__name__)

model=pickle.load(open('models/LogisticRegression.pkl','rb'))


@app.route('/')
def hello_world():
    return render_template("form.html")

@app.route('/submit', methods=['GET','POST'])
def submit():
    # MinTemp
    input_lst=[float(x) for x in request.form.values()]
    input_lst=np.array(input_lst)
    input_lst=input_lst.reshape(1,-1)
    pred = model.predict(input_lst)
    output=int(pred[0])
    print(output)
    return render_template("prediction.html",result=output)


@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)