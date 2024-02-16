# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, QuantileTransformer

import pandas as pd

# %%
data = pd.read_csv('data.csv')

data = data.dropna(axis="index" , how="any" , subset=["Price" , "Body Style" , "Fuel Efficiency" , "Engine Volume" , "Manufacture Year", "Powertrain"])

data = data.replace("Gas", "Petrol")

# Exceptional cases (Prices too high)
data = data.drop(index=1693)
data = data.drop(index=2336)
data = data.drop(index=2390)

# %%
price = data['Price']

label_encoder = LabelEncoder()

target_encoding = data.groupby('Body Style')['Price'].mean().to_dict()
data['Body Style'] = data['Body Style'].map(target_encoding)

target_encoding_2 = data.groupby('Variant')['Price'].mean().to_dict()
data['Variant'] = data['Variant'].map(target_encoding_2)

data['Gearbox'] = label_encoder.fit_transform(data['Gearbox'])
data['Manufacturer'] = label_encoder.fit_transform(data['Manufacturer'])
data['Powertrain'] = label_encoder.fit_transform(data['Powertrain'])
data['License Status'] = label_encoder.fit_transform(data['License Status'])
data['Location'] = label_encoder.fit_transform(data['Location'])
data['Owner_Type'] = label_encoder.fit_transform(data['Owner_Type'])

variables = data.drop('Price', axis=1)

# Scale variables using quantile transformer
variables = QuantileTransformer().fit_transform(variables)

# Create model object
rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=1)

param_grid = {
    'n_estimators': [180, 200, 220],
    'max_depth': [10, 12, 14]
}

model = GridSearchCV(
    estimator=rf_regressor,
    cv=5,
    scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'],
    refit='r2',
    param_grid=param_grid,
    n_jobs=-1
)

# Train the model
model.fit(variables, price)
# %%
import pickle
model_file = open('model/model.sav', 'wb')
pickle.dump(model, model_file)

bodyStyle_encoding_file = open('model/bodyStyle_encoding.sav', 'wb')
pickle.dump(target_encoding, bodyStyle_encoding_file)

variant_encoding_file = open('model/variant_encoding.sav', 'wb')
pickle.dump(target_encoding_2, variant_encoding_file)

model_file.close()
bodyStyle_encoding_file.close()
variant_encoding_file.close()

# %%
'''for key in data.keys():
    if key != 'Price':
        plt.scatter(data[key], price)
        plt.xlabel(key)
        plt.ylabel('Price')
        plt.title(f'Scatter Plot by {key}')
        plt.show()'''


