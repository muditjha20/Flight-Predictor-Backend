import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS


# load saved bundle (preprocessor + xgb + nn + feature indices + alpha)
with open("price_ensemble_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

preprocessor = bundle["preprocessor"]
xgb_model = bundle["xgb_model"]
nn_model = bundle["nn_model"]
top_indices = bundle["top_idx"]
alpha = bundle["best_alpha"]
all_features = bundle["all_features"]

# load training data to build defaults and lookup tables
train_df = pd.read_csv("flight_data_with_features (1).csv").dropna()

categorical = [c for c in all_features if train_df[c].dtype == "object"]
numerical = [c for c in all_features if c not in categorical]

defaults = {}
for col in all_features:
    if col in numerical:
        defaults[col] = float(train_df[col].mean())
    else:
        defaults[col] = train_df[col].mode()[0]

source_to_origin = train_df.groupby("Source")["origin_code"].agg(lambda x: x.mode()[0]).to_dict()
dest_to_dest = train_df.groupby("Destination")["destination_code"].agg(lambda x: x.mode()[0]).to_dict()
route_popularity_map = train_df.groupby("route_key")["route_popularity"].mean().to_dict()
metro_map = train_df.groupby("route_key")["is_metro_route"].mean().to_dict()
airline_popularity = train_df.groupby("Airline")["popularity_of_airline"].mean().to_dict()
pair_distance = train_df.groupby(["origin_code", "destination_code"])["distance"].mean().to_dict()

global_distance = defaults["distance"]
global_route_pop = defaults["route_popularity"]
global_metro = defaults["is_metro_route"]
global_airline_pop = defaults["popularity_of_airline"]


def parse_date(value):
    try:
        return pd.to_datetime(value, dayfirst=True)
    except Exception:
        return None


def parse_time(value):
    if not value:
        return None
    try:
        base = value.split()[0]
        return datetime.strptime(base, "%H:%M").time()
    except Exception:
        return None


def duration_to_minutes(value):
    if not value or not isinstance(value, str):
        return defaults["duration_minutes"]

    text = value.replace(" ", "")
    hours = 0
    minutes = 0

    if "h" in text:
        try:
            hours = int(text.split("h")[0])
            text = text.split("h")[1]
        except Exception:
            pass

    if "m" in text:
        try:
            minutes = int(text.replace("m", ""))
        except Exception:
            pass

    total = hours * 60 + minutes
    if total <= 0:
        return defaults["duration_minutes"]
    return total


def time_of_day_category(t):
    if t is None:
        return defaults["departure_time_of_day"], 0, 0

    h = t.hour
    if 5 <= h < 12:
        return "morning", 1, 0
    if 12 <= h < 17:
        return "afternoon", 0, 0
    if 17 <= h < 22:
        return "evening", 0, 1
    return "night", 0, 0


def parse_stops(value):
    if value is None:
        return defaults["num_stops"], defaults["is_direct_flight"]

    if isinstance(value, (int, float)):
        n = int(value)
        return n, 1 if n == 0 else 0

    text = str(value).lower()
    if "non" in text:
        return 0, 1

    try:
        n = int(text.split()[0])
        return n, 1 if n == 0 else 0
    except Exception:
        return defaults["num_stops"], defaults["is_direct_flight"]


def build_feature_row(payload):
    feats = defaults.copy()

    for key in ["Airline", "Source", "Destination", "Route", "Additional_Info"]:
        if key in payload:
            feats[key] = payload[key]

    src = feats["Source"]
    dst = feats["Destination"]

    feats["origin_code"] = source_to_origin.get(src, defaults["origin_code"])
    feats["destination_code"] = dest_to_dest.get(dst, defaults["destination_code"])

    d = parse_date(payload.get("Date_of_Journey"))
    if d is not None:
        feats["month"] = d.month
        feats["day_of_month"] = d.day
        feats["week_of_year"] = int(d.isocalendar().week)
        feats["quarter"] = (d.month - 1) // 3 + 1
        feats["weekday_weekend"] = 1 if d.weekday() >= 5 else 0

        month_end = (d + pd.offsets.MonthEnd(0)).day
        feats["days_until_month_end"] = month_end - d.day
        feats["is_month_start"] = 1 if d.day <= 3 else 0
        feats["is_month_end"] = 1 if d.day >= month_end - 2 else 0

    dep = parse_time(payload.get("Dep_Time"))
    arr = parse_time(payload.get("Arrival_Time"))
    dep_tod, is_morning, is_evening = time_of_day_category(dep)

    feats["departure_time_of_day"] = dep_tod
    feats["arrival_time_of_day"] = time_of_day_category(arr)[0]
    feats["is_morning_departure"] = is_morning
    feats["is_evening_departure"] = is_evening

    if dep and arr:
        feats["overnight_flight"] = 1 if arr.hour < dep.hour else 0
        feats["same_day_arrival"] = 0 if feats["overnight_flight"] else 1

    if "Duration" in payload:
        feats["duration_minutes"] = duration_to_minutes(payload["Duration"])

    num_stops, direct = parse_stops(payload.get("Total_Stops"))
    feats["num_stops"] = num_stops
    feats["is_direct_flight"] = direct

    if "route_key" in payload:
        rk = payload["route_key"]
    else:
        rk = f"{src}_{dst}"
    feats["route_key"] = rk

    feats["route_popularity"] = route_popularity_map.get(rk, global_route_pop)
    feats["is_metro_route"] = 1 if metro_map.get(rk, global_metro) >= 0.5 else 0

    feats["popularity_of_airline"] = airline_popularity.get(
        feats["Airline"], global_airline_pop
    )

    feats["is_business"] = 0
    feats["business_economy"] = 0

    oc = feats["origin_code"]
    dc = feats["destination_code"]
    feats["distance"] = pair_distance.get((oc, dc), global_distance)

    if feats["duration_minutes"] > 0:
        feats["average_speed"] = feats["distance"] / (feats["duration_minutes"] / 60)
        feats["duration_per_km"] = feats["duration_minutes"] / max(feats["distance"], 1)
    else:
        feats["average_speed"] = defaults["average_speed"]
        feats["duration_per_km"] = defaults["duration_per_km"]

    feats["is_long_haul"] = 1 if feats["distance"] > 2500 else 0
    feats["is_short_haul"] = 1 if feats["distance"] < 1200 else 0

    row = {col: feats[col] for col in all_features}
    return pd.DataFrame([row])


app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})



@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "flight price api running"})


@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    data = request.get_json()

    try:
        row = build_feature_row(data)
        X = preprocessor.transform(row)
        X_top = X[:, top_indices]

        pred_xgb = xgb_model.predict(X_top)[0]
        pred_nn = nn_model.predict(X_top)[0]
        final_pred = alpha * pred_xgb + (1 - alpha) * pred_nn

        return jsonify(
            {
                "predicted_price": float(final_pred),
                "xgb_price": float(pred_xgb),
                "nn_price": float(pred_nn),
                "alpha_xgb": float(alpha),
                "alpha_nn": float(1 - alpha),
            }
        )

    except Exception as e:
        print("prediction error:", e)
        return jsonify({"error": "prediction failed"}), 500


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
