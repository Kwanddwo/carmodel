from flask import Flask, request, jsonify
from flask_cors import CORS

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import pandas as pd
import pickle

# TODO: Better understand models and create a better (more accurate and useable here) one
# TODO: implement null and missing value handling
# TODO: Better understand dataFrames
# TODO: Learn curl

model_file = open('model/model.sav', 'rb')
model = pickle.load(model_file)
model_file.close()

bodyStyle_encoding_file = open('model/bodyStyle_encoding.sav', 'rb')
bodyStyle_encoding = pickle.load(bodyStyle_encoding_file)
bodyStyle_encoding_file.close()

variant_encoding_file = open('model/variant_encoding.sav', 'rb')
variant_encoding = pickle.load(variant_encoding_file)
variant_encoding_file.close()

app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify('hello, world')
    
    json_data = request.get_json(force=True)
    data = pd.DataFrame([json_data], index=[json_data.get('index', 0)])
    data = data.drop('index', axis=1)
    print(data)
    
    label_encoder = LabelEncoder()

    # This doesn't work, it always gives the same value, you should look up and understand how this actually works

    data['Body Style'] = data['Body Style'].map(bodyStyle_encoding)
    data['Variant'] = data['Variant'].map(variant_encoding)

    data['Gearbox'] = label_encoder.fit_transform(data['Gearbox'])
    data['Manufacturer'] = label_encoder.fit_transform(data['Manufacturer'])
    data['Powertrain'] = label_encoder.fit_transform(data['Powertrain'])
    data['License Status'] = label_encoder.fit_transform(data['License Status'])
    data['Location'] = label_encoder.fit_transform(data['Location'])
    data['Owner_Type'] = label_encoder.fit_transform(data['Owner_Type'])

    feature_values = list(data.to_numpy())  # Extract feature values as a list
    print(feature_values)
    
    prediction = model.predict(feature_values)  # Pass feature values as a list
    print(prediction)
    return jsonify({'prediction': prediction.tolist()})

@app.route('/test', methods=['GET', 'POST'])
def test():
    return jsonify({'message': 'Hello there! use POST instead'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)