import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_DIR / "flight_pricing_dataset.csv"
)


# ============================================================
# ORIGINAL DATA INFORMATION
# ============================================================

original_rows = len(df)

original_airlines = df["Airline"].nunique(dropna=True)
original_sources = df["Source"].nunique(dropna=True)
original_destinations = df["Destination"].nunique(dropna=True)

original_missing_values = df.isnull().sum().sum()


# ============================================================
# PRICE CLEANING
# ============================================================

df["Price"] = (
    df["Price"]
    .str.replace("Rs. ", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["Price"] = pd.to_numeric(
    df["Price"],
    errors="coerce"
)


# ============================================================
# DISTANCE CLEANING
# ============================================================

df["Distance_km"] = (
    df["Distance_km"]
    .str.replace(" km", "", regex=False)
)

df["Distance_km"] = pd.to_numeric(
    df["Distance_km"],
    errors="coerce"
)


# ============================================================
# DURATION CLEANING
# ============================================================

def convert_duration(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Already in minutes
    if "min" in value:
        return float(
            value.replace("min", "").strip()
        )

    # Hours and minutes
    if "h" in value:

        parts = value.replace("m", "").split("h")

        hours = float(parts[0].strip())
        minutes = float(parts[1].strip())

        return hours * 60 + minutes

    # Decimal hours
    return float(value) * 60


df["Duration"] = df["Duration"].apply(
    convert_duration
)


# ============================================================
# TOTAL STOPS CLEANING
# ============================================================

def stops(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if "non-stop" in value:
        return 0

    elif " stops" in value:
        return float(
            value.replace(" stops", "")
        )

    elif " stop" in value:
        return float(
            value.replace(" stop", "")
        )

    else:
        return float(value)


df["Total_Stops"] = df["Total_Stops"].apply(
    stops
)


# ============================================================
# PASSENGER COUNT CLEANING
# ============================================================

number_words = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6
}


def convert_passenger_count(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value in number_words:
        return number_words[value]

    elif value.isdigit():
        return int(value)

    else:
        return float(value)


df["Passenger_Count"] = df[
    "Passenger_Count"
].apply(convert_passenger_count)


# ============================================================
# AIRLINE CLEANING
# ============================================================

df["Airline"] = (
    df["Airline"]
    .str.strip()
    .str.lower()
)

cleaned_airlines = df[
    "Airline"
].nunique(dropna=True)


# ============================================================
# LOCATION CLEANING
# ============================================================

location_mapping = {
    "Ahmedabad": "Ahmedabad",
    "Ahmedabad Airport": "Ahmedabad",
    "AMD": "Ahmedabad",

    "Bangalore": "Bangalore",
    "Bangalore Airport": "Bangalore",
    "BLR": "Bangalore",

    "Bangkok": "Bangkok",
    "Bangkok Airport": "Bangkok",
    "BKK": "Bangkok",

    "Chennai": "Chennai",
    "Chennai Airport": "Chennai",
    "MAA": "Chennai",

    "Delhi": "Delhi",
    "Delhi Airport": "Delhi",
    "DEL": "Delhi",

    "Doha": "Doha",
    "Doha Airport": "Doha",
    "DOH": "Doha",

    "Dubai": "Dubai",
    "Dubai Airport": "Dubai",
    "DXB": "Dubai",

    "Frankfurt": "Frankfurt",
    "Frankfurt Airport": "Frankfurt",
    "FRA": "Frankfurt",

    "Goa": "Goa",
    "Goa Airport": "Goa",
    "GOI": "Goa",

    "Hyderabad": "Hyderabad",
    "Hyderabad Airport": "Hyderabad",
    "HYD": "Hyderabad",

    "Jaipur": "Jaipur",
    "Jaipur Airport": "Jaipur",
    "JAI": "Jaipur",

    "Kolkata": "Kolkata",
    "Kolkata Airport": "Kolkata",
    "CCU": "Kolkata",

    "London": "London",
    "London Airport": "London",
    "LHR": "London",

    "Mumbai": "Mumbai",
    "Mumbai Airport": "Mumbai",
    "BOM": "Mumbai",

    "New York": "New York",
    "New York Airport": "New York",
    "JFK": "New York",

    "Pune": "Pune",
    "Pune Airport": "Pune",
    "PNQ": "Pune",

    "Singapore": "Singapore",
    "Singapore Airport": "Singapore",
    "SIN": "Singapore",

    "Sydney": "Sydney",
    "Sydney Airport": "Sydney",
    "SYD": "Sydney",
}


df["Source"] = df["Source"].map(
    location_mapping
)

df["Destination"] = df["Destination"].map(
    location_mapping
)


cleaned_sources = df[
    "Source"
].nunique(dropna=True)

cleaned_destinations = df[
    "Destination"
].nunique(dropna=True)


# ============================================================
# DAYS BEFORE DEPARTURE
# ============================================================

df["Days_Before_Departure"] = pd.to_numeric(
    df["Days_Before_Departure"],
    errors="coerce"
)


# ============================================================
# TIME CONVERSION
# ============================================================

def convert_time(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # 12-hour format
    if "AM" in value or "PM" in value:

        parts = value.split()

        time_part = parts[0]
        period = parts[1]

        hour, minute = map(
            int,
            time_part.split(":")
        )

        if period == "AM":

            if hour == 12:
                hour = 0

        elif period == "PM":

            if hour != 12:
                hour += 12

        return hour * 60 + minute

    # 24-hour format
    else:

        hour, minute = map(
            int,
            value.split(":")
        )

        return hour * 60 + minute


df["Departure_Time"] = df[
    "Departure_Time"
].apply(convert_time)

df["Arrival_Time"] = df[
    "Arrival_Time"
].apply(convert_time)


# ============================================================
# CYCLICAL TIME FEATURES
# ============================================================

df["Departure_Time_Sin"] = np.sin(
    2 * np.pi * df["Departure_Time"] / 1440
)

df["Departure_Time_Cos"] = np.cos(
    2 * np.pi * df["Departure_Time"] / 1440
)

df["Arrival_Time_Sin"] = np.sin(
    2 * np.pi * df["Arrival_Time"] / 1440
)

df["Arrival_Time_Cos"] = np.cos(
    2 * np.pi * df["Arrival_Time"] / 1440
)


# ============================================================
# DATE FEATURES
# ============================================================

df["Departure_Date"] = pd.to_datetime(
    df["Departure_Date"],
    errors="coerce"
)

df["Departure_Year"] = (
    df["Departure_Date"].dt.year
)

df["Departure_Month"] = (
    df["Departure_Date"].dt.month
)

df["Departure_Day"] = (
    df["Departure_Date"].dt.day
)

df["Departure_DayOfYear"] = (
    df["Departure_Date"].dt.dayofyear
)


# ============================================================
# MISSING VALUES AFTER STANDARDIZATION
# ============================================================

missing_after_standardization = (
    df.isnull().sum().sum()
)


# ============================================================
# REMOVE ROWS WITH MISSING PRICE
# ============================================================

df = df.dropna(
    subset=["Price"]
)


# ============================================================
# SAVE CLEANED DATASET
# ============================================================

cleaned_file_path = DATA_DIR / "cleaned_flight_pricing_dataset.csv"

df.to_csv(
    cleaned_file_path,
    index=False
)

print(
    "Cleaned dataset saved to:",
    cleaned_file_path
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

numeric_features = [
    "Distance_km",
    "Duration",
    "Total_Stops",
    "Passenger_Count",
    "Days_Before_Departure",

    "Departure_Time_Sin",
    "Departure_Time_Cos",
    "Arrival_Time_Sin",
    "Arrival_Time_Cos",

    "Departure_Year",
    "Departure_Month",
    "Departure_Day",
    "Departure_DayOfYear"
]


categorical_features = [
    "Airline",
    "Source",
    "Destination",
    "Travel_Class",
    "Season",
    "Weekday",
    "Aircraft_Type",
    "Booking_Channel"
]


X = df[
    numeric_features + categorical_features
]

y = df["Price"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# NUMERICAL MISSING VALUES
# ============================================================

for column in numeric_features:

    median_value = X_train[
        column
    ].median()

    X_train[column] = (
        X_train[column]
        .fillna(median_value)
    )

    X_test[column] = (
        X_test[column]
        .fillna(median_value)
    )


# ============================================================
# CATEGORICAL MISSING VALUES
# ============================================================

for column in categorical_features:

    X_train[column] = (
        X_train[column]
        .fillna("Unknown")
    )

    X_test[column] = (
        X_test[column]
        .fillna("Unknown")
    )


# ============================================================
# ONE-HOT ENCODING
# ============================================================

from sklearn.preprocessing import OneHotEncoder


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


X_train_cat = encoder.fit_transform(
    X_train[categorical_features]
)

X_test_cat = encoder.transform(
    X_test[categorical_features]
)


# ============================================================
# NUMERICAL FEATURES → NUMPY
# ============================================================

X_train_num = X_train[
    numeric_features
].to_numpy()

X_test_num = X_test[
    numeric_features
].to_numpy()


# ============================================================
# COMBINE NUMERICAL + CATEGORICAL FEATURES
# ============================================================

X_train_final = np.hstack([
    X_train_num,
    X_train_cat
])

X_test_final = np.hstack([
    X_test_num,
    X_test_cat
])


# ============================================================
# SCIKIT-LEARN LINEAR REGRESSION
# ============================================================

from sklearn.linear_model import LinearRegression


model = LinearRegression()

model.fit(
    X_train_final,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predictions = model.predict(
    X_test_final
)

predictions = predictions.ravel()


# ============================================================
# MODEL METRICS
# ============================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

minimum_prediction = predictions.min()

maximum_prediction = predictions.max()

negative_predictions = (
    predictions < 0
).sum()


# ============================================================
# BASELINE
# ============================================================

baseline_predictions = np.full(
    len(y_test),
    y_train.mean()
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        baseline_predictions
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("                 DATA CLEANING SUMMARY")
print("=" * 60)

print(
    f"Original rows                 : "
    f"{original_rows}"
)

print(
    f"Rows used for modelling       : "
    f"{len(df)}"
)

print()

print("Categorical cleaning:")

print(
    f"  Airlines                    : "
    f"{original_airlines} -> {cleaned_airlines}"
)

print(
    f"  Source locations            : "
    f"{original_sources} -> {cleaned_sources}"
)

print(
    f"  Destination locations       : "
    f"{original_destinations} -> "
    f"{cleaned_destinations}"
)

print()

print("Numerical transformations:")

print(
    "  Price                       : "
    "Rs. strings -> float"
)

print(
    "  Distance_km                 : "
    "'km' strings -> float"
)

print(
    "  Duration                    : "
    "mixed formats -> minutes"
)

print(
    "  Total_Stops                 : "
    "text -> number"
)

print(
    "  Passenger_Count             : "
    "words/numbers -> float"
)

print(
    "  Days_Before_Departure       : "
    "strings -> float"
)

print(
    "  Departure_Time              : "
    "mixed formats -> minutes"
)

print(
    "  Arrival_Time                : "
    "mixed formats -> minutes"
)

print()

print("Missing values:")

print(
    f"  Original dataset             : "
    f"{original_missing_values}"
)

print(
    f"  After standardization        : "
    f"{missing_after_standardization}"
)


print()
print("=" * 60)
print("                 MODEL PERFORMANCE")
print("=" * 60)

print(
    f"MSE                           : "
    f"{mse:.2f}"
)

print(
    f"RMSE                          : "
    f"{rmse:.2f}"
)

print(
    f"MAE                           : "
    f"{mae:.2f}"
)

print(
    f"R²                            : "
    f"{r2:.4f}"
)

print(
    f"Minimum prediction            : "
    f"{minimum_prediction:.2f}"
)

print(
    f"Maximum prediction            : "
    f"{maximum_prediction:.2f}"
)

print(
    f"Negative predictions          : "
    f"{negative_predictions}"
)

print(
    f"Baseline RMSE                 : "
    f"{baseline_rmse:.2f}"
)

print("=" * 60)

print()
print("First 10 predictions:")

print(predictions[:10])
