from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

#load data
df_r = pd.read_csv("./wineData/winequality-red.csv", sep = ';')
data_r = df_r.values

#Oversample the rare labels
strategy_r = {3.0: 50, 8.0: 50}
OverSample_random = RandomOverSampler(sampling_strategy = strategy_r)
X_res, y_res = OverSample_random.fit_resample(data_r[:,:-1], data_r[:,-1])

#train the model
RF = RandomForestClassifier(n_estimators=500)
RF.fit(X_res, y_res)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    data = data.values
    #make predicon using model
    prediction = RF.predict(data)
    return Response(json.dumps(prediction[0]))

