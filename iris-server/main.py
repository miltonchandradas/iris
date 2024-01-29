import os
import pickle
import numpy as np
from flask import Flask, render_template, request, redirect

# Creates Flask serving engine
app = Flask(__name__)

model = None
appHasRunBefore = False
flower = ""
sepal_length = ""
sepal_width = ""
petal_length = ""
petal_width = ""

@app.before_request
def init():
    """
    Load model else crash, deployment will not start
    """
    global model
    global appHasRunBefore
    
    if not appHasRunBefore:
        model = pickle.load(open ('/mnt/models/model.pkl','rb')) # All the model files will be read from /mnt/models
        appHasRunBefore = True
        return None

@app.route("/v2/greet", methods=["GET"])
def status():
    global model
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        return render_template("index.html")
    
    
@app.route("/v2/predict", methods=["GET", "POST"])
def predict():
    global model
    global flower
    global petal_length
    global petal_width
    global sepal_length
    global sepal_width
    
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        if request.method == "POST":
            sepal_length = request.form["SepalLengthCm"]
            sepal_width = request.form["SepalWidthCm"]
            petal_length = request.form["PetalLengthCm"]
            petal_width = request.form["PetalWidthCm"]
            attributes = [sepal_length, sepal_width, petal_length, petal_width]            
            prediction = model.predict(
                np.array([list(map(float, attributes)),]) # (trailing comma) <,> to make batch with 1 observation
            )
            if str(prediction) == "[0]":
                flower = "Setosa"
            elif str(prediction) == "[1]":
                flower = "Veriscolor"
            elif str(prediction) == "[2]":
                flower ="Virginica"
            else: 
                flower = "Unkonwn"
            
            return redirect("/v2/predict")
        else:
            return render_template("index.html", flower=flower, sepal_length = sepal_length, sepal_width = sepal_width, petal_length = petal_length, petal_width = petal_width)

if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print("Serving Started")
    app.run(host="0.0.0.0", debug=True, port=9001)
