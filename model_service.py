'''
Flask API fo Model Interaction
'''
from flask import Flask, jsonify, request
from flasgger import Swagger
import pandas as pd



app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the sentiment of the restaurant review
    '''
    pass



@app.route('/load', methods=['POST'])
def load():
    '''
    Load specific model
    '''
    pass