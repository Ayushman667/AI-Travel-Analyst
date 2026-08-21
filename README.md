# AI Travel Analyst

A machine-learning project for predicting flight ticket prices from a real-world flight-pricing dataset.

The project focuses on the complete workflow:

**raw data → data cleaning → feature engineering → train/test split → encoding → linear regression → evaluation**

Two implementations of multiple linear regression are included:

1. **Custom implementation** using NumPy and gradient descent.
2. **Scikit-learn implementation** using `LinearRegression`.

The two models use the same cleaned data, feature definitions, train/test split, missing-value handling, and one-hot encoding so that their results can be compared fairly.

---

## Project Structure

```text
AI-Travel-Analyst/
│
├── data/
│   ├── flight_pricing_dataset.csv
│   └── cleaned_flight_pricing_dataset.csv
│
├── models/
│   ├── my_linear_regression.py
│   └── sklearn_model.py
│
├── notebooks/
│   └── flight_analysis.ipynb
│
├── src/
│   └── preprocessing.py
│
├── README.md
└── requirements.txt
```

---

## Dataset

The original dataset contains **100,000 flight records**.

The target variable is:

```text
Price
```

The dataset contains numerical and categorical information about flights, including:

- Airline
- Source
- Destination
- Travel Class
- Season
- Weekday
- Aircraft Type
- Booking Channel
- Distance
- Duration
- Number of Stops
- Passenger Count
- Days Before Departure
- Departure Time
- Arrival Time
- Departure Date

---

## 1. Data Cleaning

The original dataset contains inconsistent representations and missing values.

### Price

Values such as:

```text
Rs. 45,000
```

are converted into numerical values:

```text
45000.0
```

### Distance

Values containing `"km"` are converted into floating-point numbers.

### Duration

Duration appears in several formats, including:

```text
177 min
3h 11m
1.67
```

All duration values are converted into **minutes**.

### Total Stops

Different textual representations such as:

```text
non-stop
1 stop
2 stops
```

are converted into numerical values:

```text
0
1
2
```

### Passenger Count

Passenger counts written as words or numbers are converted into numerical values.

For example:

```text
"three" → 3
"5"     → 5
```

### Airline

Airline names are stripped of surrounding whitespace and converted to lowercase so that capitalization differences do not create unnecessary categories.

### Source and Destination

The dataset contains multiple representations of the same location.

For example:

```text
Ahmedabad
Ahmedabad Airport
AMD
```

are mapped to:

```text
Ahmedabad
```

The same mapping is applied to both Source and Destination.

This reduces the number of unique raw location representations from **54 to 18** for both Source and Destination.

### Days Before Departure

The column is converted from strings to numerical values.

### Time

Departure and arrival times can appear in both 12-hour and 24-hour formats.

They are converted into minutes after midnight.

Time is then represented using cyclical features:

```text
Departure_Time_Sin
Departure_Time_Cos
Arrival_Time_Sin
Arrival_Time_Cos
```

This is important because time is cyclical: for example, 23:59 and 00:00 are close in actual time but far apart numerically.

### Date

Departure dates are converted to datetime values and additional features are extracted:

```text
Departure_Year
Departure_Month
Departure_Day
Departure_DayOfYear
```

### Missing Target Values

Rows where `Price` is missing are removed because the model cannot learn from a training example without a target value.

---

## 2. Feature Preparation

The model uses the following numerical features:

```text
Distance_km
Duration
Total_Stops
Passenger_Count
Days_Before_Departure
Departure_Time_Sin
Departure_Time_Cos
Arrival_Time_Sin
Arrival_Time_Cos
Departure_Year
Departure_Month
Departure_Day
Departure_DayOfYear
```

Categorical features are:

```text
Airline
Source
Destination
Travel_Class
Season
Weekday
Aircraft_Type
Booking_Channel
```

The categorical features cannot be passed directly to the linear regression model because they contain strings.

They are therefore converted using **one-hot encoding**.

Unknown categorical values are handled using:

```python
handle_unknown="ignore"
```

The final feature matrix contains both the numerical features and the one-hot encoded categorical features.

---

## 3. Train/Test Split

The cleaned dataset is divided into:

- **80% training data**
- **20% testing data**

The split uses:

```python
random_state=42
```

This makes the split reproducible.

Missing numerical feature values are filled using the **median calculated from the training data**.

Categorical missing values are replaced with:

```text
Unknown
```

The encoder is fitted only on the training categorical data and then used to transform the test data.

This prevents information from the test set from influencing the encoding process.

---

# 4. Custom Linear Regression

`models/my_linear_regression.py` contains a custom implementation of linear regression.

The model does not use Scikit-learn's regression estimator.

Instead, the mathematics is implemented using NumPy.

The prediction equation is:

```text
ŷ = Xw + b
```

where:

- `X` = feature matrix
- `w` = weight vector
- `b` = bias
- `ŷ` = predicted price

### Feature Standardization

Before gradient descent, the features are standardized:

```text
X_scaled = (X - mean) / standard_deviation
```

The mean and standard deviation are learned from the training data and stored in the model.

A zero standard deviation is replaced with `1` to avoid division by zero.

### Gradient Descent

For each epoch, the model calculates:

```text
prediction = Xw + b
error = prediction - y
```

The gradients are:

```text
dw = Xᵀ(error) / m
db = mean(error)
```

The parameters are then updated:

```text
w = w - learning_rate × dw
b = b - learning_rate × db
```

The model stores the mean squared error after every epoch so that the training-cost curve can be plotted.

The current implementation uses:

```text
learning_rate = 0.01
epochs = 500
```

---

# 5. Scikit-learn Linear Regression

`models/sklearn_model.py` performs the same overall preprocessing and uses:

```python
from sklearn.linear_model import LinearRegression
```

The model is created with:

```python
model = LinearRegression()
```

and trained using:

```python
model.fit(X_train_final, y_train)
```

Unlike the custom implementation, the optimization mathematics is handled internally by Scikit-learn.

This makes the Scikit-learn implementation much faster and more concise.

---

# 6. Model Evaluation

The models are evaluated on the **test set**, not the training set.

The project calculates:

### MSE — Mean Squared Error

Measures the average squared prediction error.

```text
MSE = mean((y - ŷ)²)
```

Lower is better.

### RMSE — Root Mean Squared Error

The square root of MSE.

```text
RMSE = √MSE
```

It is easier to interpret because it is in the same units as the target price.

Lower is better.

### MAE — Mean Absolute Error

Measures the average absolute difference between the predicted and actual prices.

```text
MAE = mean(|y - ŷ|)
```

Lower is better.

### R² — Coefficient of Determination

Measures how much of the variation in the target is explained by the model.

A value closer to `1` indicates better explanatory performance.

A value around `0` means the model explains little more variation than a simple mean-based prediction.

### Baseline RMSE

The project also calculates a baseline model that predicts the mean training price for every test example.

The linear regression model should be compared against this baseline rather than judging its performance in isolation.

### Negative Predictions

The project counts predictions below zero.

A negative flight price is not physically meaningful, so this is an important limitation of ordinary linear regression for this target.

The current linear regression model produces some negative predictions.

They should **not** simply be multiplied by `-1`, because that would arbitrarily change the model output rather than fixing the underlying model.

---

# 7. Observed Results

On the current dataset/run, the custom implementation produced approximately:

```text
MSE  : 2.728 × 10^9
RMSE : 52,231
MAE  : 23,471
R²   : 0.5508
```

The Scikit-learn implementation produced approximately:

```text
RMSE : 52,152
MAE  : 23,424
R²   : 0.5522
```

The exact values can vary if the preprocessing or dataset changes.

The two implementations therefore produce **very similar predictive performance**.

This is useful because it provides a practical comparison between:

```text
understanding and implementing the mathematics yourself
```

and:

```text
using a production-ready machine-learning library
```

---

# 8. Training Cost Visualization

The custom implementation stores the cost after each gradient-descent iteration.

The project plots:

```text
Epoch → Cost
```

This allows the training process to be visually inspected.

A decreasing curve indicates that gradient descent is reducing the training loss.

The curve flattening means that additional epochs are producing progressively smaller improvements.

---

# 9. Data Analysis Notebook

`notebooks/flight_analysis.ipynb` is intended for exploratory data analysis (EDA).

It can be used to investigate:

- Dataset structure
- Missing values
- Unique categories
- Numerical distributions
- Price statistics
- Feature relationships
- Correlations
- Data-cleaning decisions
- Model-related observations

The notebook is separate from the model scripts so that exploratory analysis does not get mixed with the final training pipeline.

---

# 10. Why Compare Two Implementations?

The custom implementation demonstrates the underlying mathematics of linear regression and gradient descent.

The Scikit-learn implementation demonstrates how the same problem is solved using a standard machine-learning library.

The comparison helps answer questions such as:

- Does my implementation produce reasonable results?
- Does gradient descent actually converge?
- How close are the results to a standard implementation?
- How much faster is the optimized library implementation?
- What limitations does linear regression have on this real-world dataset?

---

# 11. Limitations

Linear regression assumes that the target can be reasonably modeled as a linear combination of the input features.

Flight pricing is more complicated than that.

Price can depend on nonlinear interactions between:

- Airline
- Route
- Travel class
- Demand
- Timing
- Number of stops
- Booking lead time
- Season
- Other hidden factors

Therefore, linear regression is useful as a **baseline and learning model**, but it may not be the optimal model for production-level flight-price prediction.

The negative predictions are another limitation of unconstrained linear regression because the model has no built-in understanding that a price cannot be negative.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## Running the Project

From the project directory:

```bash
pip install -r requirements.txt
```

Run the custom implementation:

```bash
python models/my_linear_regression.py
```

Run the Scikit-learn implementation:

```bash
python models/sklearn_model.py
```

Open the analysis notebook:

```bash
jupyter notebook notebooks/flight_analysis.ipynb
```

---

## Project Goal

The primary goal of this project is not simply to obtain the lowest possible error.

It is to understand the complete machine-learning workflow:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Feature Engineering
     ↓
Missing-Value Handling
     ↓
Categorical Encoding
     ↓
Train/Test Split
     ↓
Feature Scaling
     ↓
Linear Regression
     ↓
Gradient Descent
     ↓
Predictions
     ↓
Model Evaluation
     ↓
Comparison with Scikit-learn
```

