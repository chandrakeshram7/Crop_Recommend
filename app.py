from flask import Flask,  jsonify
import requests
import config
import pickle
import io
import boto3
from io import BytesIO
from flask_cors import CORS
import config
import urllib3
from urllib3.util.ssl_ import DEFAULT_CIPHERS
import os
aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
bucket_name = 'mymlmodel1'
model_key = 'DecisionTree.pkl'

response = s3.get_object(Bucket=bucket_name, Key=model_key)
model_bytes = response['Body'].read()

crop_recommendation_model = pickle.loads(model_bytes)
app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])



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
    
    data = [[N, P, K, temperature, humidity, ph, rainfall]]
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]

    return jsonify({"prediction" : final_prediction})

from app import app
if __name__ == '__main__':
    app.run(debug=True, port=5000)

#lets check