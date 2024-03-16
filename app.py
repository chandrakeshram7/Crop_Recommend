# Importing essential libraries and modules
from flask import Flask,  jsonify
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import boto3
from io import BytesIO
from flask_cors import CORS
import config
s3 = boto3.client('s3')
bucket_name = 'mymlmodel1'
model_key = 'DecisionTree.pkl'

# Load the pickled model file from S3 directly into memory
response = s3.get_object(Bucket=bucket_name, Key=model_key)
model_bytes = response['Body'].read()

# Load the model from the bytes object
model = pickle.load(BytesIO(model_bytes))
crop_recommendation_model = pickle.loads(model_bytes)
# =========================================================================================

app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])
# render home page


@ app.route('/')
def home():
    return "Server is running, on PORT - 5000"


@ app.route('/crop-predict')
def crop_prediction():
    
    N = 40
    P = 45
    K = 50
    ph = 6
    rainfall =231
    temperature = 27.89
    humidity = 21.232
    
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]

    return jsonify({"prediction" : final_prediction})



