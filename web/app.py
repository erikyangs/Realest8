from flask import Flask, request, render_template
# Import Python Packages
import re
import warnings

# Import Standard ML packages
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template("main.html")
    elif request.method == 'POST':
        longitude = float(request.form["longitude"])
        latitude = float(request.form["latitude"])
        accomodates = float(request.form["accomodates"])
        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        beds = float(request.form["beds"])
        room_type = request.form["room_type"]
        neighbourhood_cleansed = request.form["neighbourhood_cleansed"]
        zipcode = request.form["zipcode"] + ".0"

        predicted_price = predict_price(
            longitude,
            latitude,
            accomodates,
            bedrooms,
            bathrooms,
            beds,
            room_type,
            neighbourhood_cleansed,
            zipcode,
        )

        predicted_trend = predict_trend(
            predicted_price,
            longitude,
            latitude,
            accomodates,
            bedrooms,
            bathrooms,
            beds,
            room_type,
            neighbourhood_cleansed,
            zipcode,
        )

        return render_template(
            "results.html",
            longitude = longitude,
            latitude = latitude,
            accomodates = accomodates,
            bedrooms = bedrooms,
            bathrooms = bathrooms,
            beds = beds,
            room_type = room_type,
            neighbourhood_cleansed = neighbourhood_cleansed,
            zipcode = zipcode[:-2],
            predicted_price = np.round(predicted_price, 2),
            predicted_trend = np.round(predicted_trend * 100, 2),
        )

if __name__ == '__main__':
    app.run(debug=True)

# ======= From Pipeline Notebook =======
# Import Models
import_path = "../exported_models/airbnb_price_predictor.hdf"
predictor_X_df = pd.read_hdf(import_path, "X_df")
predictor_Y = pd.read_hdf(import_path, "Y")

import_path = "../exported_models/airbnb_price_trends.hdf"
trend_X_df = pd.read_hdf(import_path, "X_df")
trend_Y = pd.read_hdf(import_path, "Y")

import_path = "../exported_models/airbnb_price_predictor.pkl"
price_model = joblib.load(import_path)

import_path = "../exported_models/airbnb_price_trends.pkl"
trend_model = joblib.load(import_path)

# Setups
def scale_X(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X)

def cols_start_with(df, s):
    cols = df.columns
    cols = cols[cols.str.startswith(s)].values
    return list(cols)

def col_suffixes(df, s):
    cols = cols_start_with(df, s)
    cols = [re.sub(f"""{s}""", "", c) for c in cols]
    return cols

def val_one_hot_transform(df, val, col_prefix):
    cols = col_suffixes(df, col_prefix)
    result = pd.DataFrame()
    for c in cols:
        c_name = col_prefix + c
        if c == val:
            result[c_name] = [1]
        else:
            result[c_name] = [0]
    return result

def predict_price_input_to_X_df(
    longitude,
    latitude,
    accomodates,
    bedrooms,
    bathrooms,
    beds,
    room_type,
    neighbourhood_cleansed,
    zipcode
):
    X_df = pd.DataFrame()
    X_df["longitude"] = [longitude]
    X_df["latitude"] = [latitude]
    X_df["accomodates"] = [accomodates]
    X_df["bedrooms"] = [bedrooms]
    X_df["bathrooms"] = [bathrooms]
    X_df["beds"] = [beds]

    X_df = pd.concat((X_df, val_one_hot_transform(predictor_X_df, room_type, "room_type_")), axis=1)
    X_df = pd.concat((X_df, val_one_hot_transform(predictor_X_df, neighbourhood_cleansed, "neighbourhood_cleansed_")), axis=1)
    X_df = pd.concat((X_df, val_one_hot_transform(predictor_X_df, zipcode, "zipcode_")), axis=1)

    return X_df

def predict_price(
    longitude,
    latitude,
    accomodates,
    bedrooms,
    bathrooms,
    beds,
    room_type,
    neighbourhood_cleansed,
    zipcode
):
    X_df = predict_price_input_to_X_df(
        longitude,
        latitude,
        accomodates,
        bedrooms,
        bathrooms,
        beds,
        room_type,
        neighbourhood_cleansed,
        zipcode
    )
    X = X_df.pipe(scale_X)
    return price_model.predict(X)[0]

def predict_trend_input_to_X_df(
    price_past,
    longitude,
    latitude,
    accomodates,
    bedrooms,
    bathrooms,
    beds,
    room_type,
    neighbourhood_cleansed,
    zipcode
):
    X_df = pd.DataFrame()
    X_df["price_past"] = [price_past]
    X_df["longitude"] = [longitude]
    X_df["latitude"] = [latitude]
    X_df["accomodates"] = [accomodates]
    X_df["bedrooms"] = [bedrooms]
    X_df["bathrooms"] = [bathrooms]
    X_df["beds"] = [beds]

    X_df = pd.concat((X_df, val_one_hot_transform(trend_X_df, room_type, "room_type_")), axis=1)
    X_df = pd.concat((X_df, val_one_hot_transform(trend_X_df, neighbourhood_cleansed, "neighbourhood_cleansed_")), axis=1)
    X_df = pd.concat((X_df, val_one_hot_transform(trend_X_df, zipcode, "zipcode_")), axis=1)

    return X_df

def predict_trend(
    price_past,
    longitude,
    latitude,
    accomodates,
    bedrooms,
    bathrooms,
    beds,
    room_type,
    neighbourhood_cleansed,
    zipcode
):
    X_df = predict_trend_input_to_X_df(
        price_past,
        longitude,
        latitude,
        accomodates,
        bedrooms,
        bathrooms,
        beds,
        room_type,
        neighbourhood_cleansed,
        zipcode
    )
    X = X_df.pipe(scale_X)
    return trend_model.predict(X)[0]
