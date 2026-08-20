import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("multiple_linear_regression/AI-Travel-Analyst/data/flight_pricing_dataset.csv")

# print(df.info())

# print(df.isnull().sum())

# print(df.nunique())

# print(df["Airline"].unique())

# print(df["Total_Stops"].unique())

# print(df["Distance_km"].head(10))

# print(df["Duration"].head(10))

# print(df["Price"].head(10))

# print(df["Duration"].dropna().head(30).to_string())

# print(df["Duration"].dropna().tail(30).to_string())

# print(df["Price"].describe())

# by running the above codes, we can see that:
# Problem                     Example
# ────────────────────────────────────────────
# Missing values              NaN
# Inconsistent capitalization Indigo / INDIGO / indigo
# Inconsistent stops          1 / 1 stop
# Mixed duration formats      1.67 / 3h 11m / 177 min
# Numeric columns as strings  Price / Distance / Passenger_Count
# Potential price cap         200000 occurs 8,989 times

### This part of the code is to find the things that are different in all of the features, and then convert all to a single format, so that we can use them for our model training and testing.
# print(df[df["Duration"].str.match(r"^\d+\.\d+$", na=False)][
#     ["Duration", "Distance_km", "Price"]
# ].head(20).to_string(index=False))

# print(
#     df[df["Duration"].str.contains("min", na=False)][
#         ["Duration", "Distance_km", "Price"]
#     ].head(20).to_string(index=False)
# )


### converting the price section to a single format of float, so that we can use it for our model training and testing.
df["Price"] = (
    df["Price"]
    .str.replace("Rs. ", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["Price"] = pd.to_numeric(df["Price"], errors="coerce")


### now to standerdise the distance feature, we will convert all the values to floating numbers

df["Distance_km"] = (
    df["Distance_km"]
    .str.replace(" km", "", regex=False)
)

df["Distance_km"] = pd.to_numeric(
    df["Distance_km"],
    errors="coerce"
)



### now to convert the duration feature to a single format, we will convert all of them in minutes:

def convert_duration(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Case 1: value is already in minutes
    if "min" in value:
        return float(value.replace("min", "").strip())

    # Case 2: value is in hours and minutes
    if "h" in value:
        parts = value.replace("m", "").split("h")

        hours = float(parts[0].strip())
        minutes = float(parts[1].strip())

        return hours * 60 + minutes

    # Case 3: decimal hours
    return float(value) * 60


df["Duration"] = df["Duration"].apply(convert_duration)


### now to converting the stops feature:

def stops(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # Case 1: value contains "non-stop":
    if "non-stop" in value:
        return 0

    # Case 2: value contains "1 stop":
    elif " stops" in value:
        return float(value.replace(" stops", ""))
    
    # Case 3: value contains "2 stops":
    elif " stop" in value:
        return float(value.replace(" stop", ""))

    else:
        return float(value)


df["Total_Stops"] = df["Total_Stops"].apply(stops)

# print(df["Total_Stops"].isnull().sum())   ** this is to check if we created any null values in the Total_Stops feature after converting it to a single format.



### now to convert the passanger count to a single format:

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

    # Convert number words to digits
    if value in number_words:
        return number_words[value]

    elif value.isdigit():
        return int(value)

    else:
        return float(value)


df["Passenger_Count"] = df["Passenger_Count"].apply(convert_passenger_count)



### now to see the airline feature:
# print(df["Airline"].unique())

# the above line of code shows that we have 40 unique types of airlines, but most of them are just capitalized varients of the same airline, so i will convert all of them to lowercase

df["Airline"] = df["Airline"].str.strip().str.lower()

# this leaves us with 13 unique airlines.

### Now i am going to see all the other features all at once

# print(df["Source"].unique())
# print(df["Destination"].unique())
# print(df["Travel_Class"].unique())
# print(df["Season"].unique())
# print(df["Weekday"].unique())
# print(df["Aircraft_Type"].unique())
# print(df["Booking_Channel"].unique())


# print(df["Source"].nunique())
# print(df["Destination"].nunique())

# print(sorted(df["Source"].dropna().unique()))

# print(sorted(df["Destination"].dropna().unique()))

# the above line of code shows that Source and Destination contain the same 54 raw location representations, so i will make a single dictionary and apply it to both

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



df["Source"] = df["Source"].map(location_mapping)
df["Destination"] = df["Destination"].map(location_mapping)

# print(len(location_mapping))             # tells the number of raw location names

# print(df["Source"].nunique())            # tells the number of unique location names if source
# print(df["Destination"].nunique())       # tells the number of unique location names if destination


### now i am checking the remaining columns and the values that it contains

# print(df["Departure_Date"].head(20))
# print(df["Departure_Time"].head(20))
# print(df["Arrival_Time"].head(20))
# print(df["Days_Before_Departure"].head(20))

# print(df["Departure_Date"].unique()[:20])
# print(df["Departure_Time"].unique()[:20])
# print(df["Arrival_Time"].unique()[:20])
# print(df["Days_Before_Departure"].unique()[:20])



### changing the days_before_departure to a consistent format:

df["Days_Before_Departure"] = pd.to_numeric(
    df["Days_Before_Departure"],
    errors="coerce"
)

# print(df["Days_Before_Departure"].head(20))
# print(df["Days_Before_Departure"].dtype)


### now converting the timings:-


def convert_time(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    # 12-hour format
    if "AM" in value or "PM" in value:
        parts = value.split()

        time_part = parts[0]
        period = parts[1]

        hour, minute = map(int, time_part.split(":"))

        if period == "AM":
            if hour == 12:
                hour = 0

        elif period == "PM":
            if hour != 12:
                hour += 12

        return hour * 60 + minute

    # 24-hour format
    else:
        hour, minute = map(int, value.split(":"))
        return hour * 60 + minute

df["Departure_Time"] = df["Departure_Time"].apply(convert_time)
df["Arrival_Time"] = df["Arrival_Time"].apply(convert_time)

print(df[["Departure_Time", "Arrival_Time"]].head(20))
print(df["Departure_Time"].dtype)
print(df["Arrival_Time"].dtype)


print(df[
    [
        "Price",
        "Distance_km",
        "Duration",
        "Total_Stops",
        "Passenger_Count",
        "Days_Before_Departure",
        "Departure_Time",
        "Arrival_Time"
    ]
].info())

print(df.isnull().sum())





class MyLinearRegression:
    def __init__(
            self,
            learning_rate = 0.01,
            epochs = 1000
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        self.costs = []
        # initialising the weights and bias

        n_features = X.shape[1]
        self.w = np.zeros((n_features, 1))

        # optimization:

        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)

        X = (X - self.mean) / self.std

        # repeting the process for number of epochs
        for _ in range(self.epochs):

            # Gradient Descent
            prediction = X @ self.w + self.b
            error = prediction - y
            dw = X.T @ error / X.shape[0]
            db = np.mean(error)

            # updation of weights and bias:
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            cost = np.mean(error ** 2)

            self.costs.append(cost)

    def predict(self, X):
        if self.w is None:
            raise ValueError("Model is not fitted yet. Please call the fit method before predicting.")
        X = (X - self.mean) / self.std
        return X @ self.w + self.b

