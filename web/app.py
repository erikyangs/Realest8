from flask import Flask, request, render_template, Response
# Import Python Packages
import re
import warnings
import io

# Import Standard ML packages
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

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

import_path = "../exported_models/airbnb_rent_comparison.hdf"
jan17_df = pd.read_hdf(import_path, "jan17_df")
may18_df = pd.read_hdf(import_path, "may18_df")
aug18_df = pd.read_hdf(import_path, "aug18_df")
nov18_df = pd.read_hdf(import_path, "nov18_df")

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

def nbr_in_rent(c):
    return nov18_df.index.isin([c]).any()

def visualize_rent_history(nbr, num_bed):
    if not nbr_in_rent(nbr):
        warnings.warn("Rent prices unavailable for this neighbourhood.")
        return None

    beds_map = {
        "0": "Studio",
        "1": "1 Bed",
        "2": "2 Beds",
        "3": "3 Beds"
    }
    bed_col = beds_map[num_bed]

    dates = [1, 12+5, 12+8, 12+11]
    rent_prices = [
        jan17_df.loc[nbr, bed_col],
        may18_df.loc[nbr, bed_col],
        aug18_df.loc[nbr, bed_col],
        nov18_df.loc[nbr, bed_col]
    ]

    plt.clf()
    plt.title(f"""Historical Data for {bed_col} Apartment in {nbr}""")
    plt.xticks(dates, ["Jan '17", "May '18", "Aug '18", "Nov '18"])
    plt.ylabel("Price/Month ($)")
    plt.plot(dates, rent_prices, 'o-')
    #plt.savefig('../plots/Airbnb Rent Comparison Sample Historical Rent.png', bbox_inches='tight')
    return plt.gcf()

def get_rent_price(nbr, num_bed):
    if not nbr_in_rent(nbr):
        warnings.warn("Rent prices unavailable for this neighbourhood.")
        return None

    beds_map = {
        "0": "Studio",
        "1": "1 Bed",
        "2": "2 Beds",
        "3": "3 Beds"
    }
    bed_col = beds_map[num_bed]
    return nov18_df.loc[nbr, bed_col]

def price_comparison_stats(nbr, num_beds, airbnb_price):
    if not nbr_in_rent(nbr):
        warnings.warn("Rent prices unavailable for this neighbourhood.")
        return None

    airbnb_price = float(airbnb_price)

    rent_price = get_rent_price(nbr, num_beds)
    breakeven = rent_price/airbnb_price

    return {
        "airbnb_price": airbnb_price,
        "nbr": nbr,
        "num_bed": num_beds,
        "rent_price": rent_price,
        "breakeven_days": breakeven,
        "breakeven_ratio": breakeven/30
    }

def visualize_stats(stats):
    if stats == None or len(stats) != 6:
        warnings.warn("Stats dont have right properties")
        return None

    plt.clf()
    plt.figure(figsize=(8,8))
    plt.suptitle("Airbnb vs. Rent Prices")

    plt.subplot(2, 1, 1)
    plt.title("Price Comparison")
    plt.ylabel("Daily Price ($)")
    plt.xticks(np.arange(2), ["Average Rent", "Airbnb"])
    prices = [stats["rent_price"]/30, stats["airbnb_price"]]
    plt.bar(np.arange(2), prices)

    plt.subplot(2, 1, 2)
    plt.title("Breakeven Point")
    days = np.arange(31)
    price_day = days * stats["airbnb_price"]
    plt.xlabel("Days Airbnb is Rented")
    plt.ylabel("Cumulative Price")
    plt.axvline(x=stats["breakeven_days"], color="r")
    plt.text(
        stats["breakeven_days"]-1.05,
        25*stats["airbnb_price"],
        f"""Breakeven = {np.round(stats["breakeven_days"], 1)} days""",
        rotation=90,
        fontsize=14
    )
    plt.plot(days, price_day)
    #plt.savefig('../plots/Airbnb Rent Comparison Sample.png', bbox_inches='tight')
    return plt.gcf()

# ========= Flask App ===========
app = Flask(__name__)

@app.route('/viz_1/<nbr>/<beds>')
def viz_1(nbr, beds):
    print(nbr, beds)
    fig = visualize_rent_history(nbr, beds)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/viz_2/<nbr>/<beds>/<predicted_price>')
def viz_2(nbr, beds, predicted_price):
    fig = visualize_stats(price_comparison_stats(nbr, beds, predicted_price))
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

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
            viz_1_url = f"""/viz_1/{neighbourhood_cleansed}/{str(int(bedrooms))}""",
            viz_2_url = f"""/viz_2/{neighbourhood_cleansed}/{str(int(bedrooms))}/{predicted_price}"""
        )

if __name__ == '__main__':
    app.run()
