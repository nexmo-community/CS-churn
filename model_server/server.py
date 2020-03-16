from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import joblib
import pandas as pd
import random
import os
from dotenv import load_dotenv
load_dotenv()
PORT = os.getenv("PORT")

app = Flask(__name__)
cors = CORS(app)

def random_bool():
    return random_number()

def random_number(low=0, high=1):
    return random.randint(low,high)

def generate_data():
    internetServices = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    paymentMethods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']

    random_data = {
            'name':'customer',
            'Partner': random_bool(),
            'Dependents': random_bool(),
            'tenure': random_number(0,50),
            'PhoneService': random_bool(),
            'MultipleLines': random_number(-1),
            'InternetService': random.choice(internetServices),
            'OnlineSecurity': random_number(-1),
            'OnlineBackup': random_number(-1),
            'DeviceProtection': random_number(-1),
            'TechSupport': random_number(-1),
            'StreamingTV': random_number(-1),
            'StreamingMovies': random_number(-1),
            'Contract': random.choice(contracts),
            'PaperlessBilling': random_bool(),
            'PaymentMethod': random.choice(paymentMethods)
        }
    return random_data

@app.route('/predict', methods=['GET'])
@cross_origin()
def predict():

    random_user_data = generate_data()
    #https://towardsdatascience.com/a-flask-api-for-serving-scikit-learn-models-c8bcdaa41daa
    query = pd.get_dummies(pd.DataFrame(random_user_data, index=[0]))
    query = query.reindex(columns=model_columns, fill_value=0)

    #return prediction as probability in percent
    prediction = round(model.predict_proba(query)[:,1][0], 2)* 100
    return jsonify({'churn': prediction})

if __name__ == '__main__':
     model = joblib.load('model/model.pkl')
     model_columns = joblib.load('model/model_columns.pkl')
     app.run(port=PORT)
