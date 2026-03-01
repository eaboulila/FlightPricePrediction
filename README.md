# ✈️ Flight Price Prediction – End-to-End Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-black)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-red)
![Status](https://img.shields.io/badge/Status-Production--Ready-success)

> From raw flight data → EDA → Feature Engineering → Model Training → Evaluation → Deployment-ready pipeline.

# 📌 Business Problem

Airline ticket prices fluctuate based on:

* Airline brand
* Number of stops
* Flight duration
* Time of departure
* Seasonal demand

The objective is to build a **robust regression model** that predicts flight ticket prices with high accuracy.

# 🔎 1️⃣ Exploratory Data Analysis (EDA)

## Dataset Overview

The dataset includes:

* Airline
* Source
* Destination
* Date of Journey
* Duration
* Total Stops
* Additional Info
* Price (Target)

## Key Insights

* ✈️ Non-stop flights generally cost more.
* 🕒 Longer duration flights are often cheaper.
* 📅 Month & season significantly impact pricing.
* 🏷 Premium airlines maintain higher base fares.
* 🌙 Early departures can influence ticket cost.

# 🧠 2️⃣ Feature Engineering

Engineered features include:

* Journey Month
* Journey Day
* Departure Hour
* Arrival Hour
* Duration in Minutes
* Weekend Indicator
* Peak Season Flag

Categorical Encoding:

* One-Hot Encoding (Nominal)
* Ordinal Encoding (Stops)

Outlier Handling:

* IQR-based filtering
* Log transformation on price (optional)

# 🏗 3️⃣ Machine Learning Pipeline

Implemented using `sklearn Pipeline`:

```python
Pipeline([
    ('preprocessing', ColumnTransformer(...)),
    ('model', XGBRegressor())
])
```

Pipeline handles:

* Missing values
* Encoding
* Scaling (if needed)
* Model training
* Cross-validation

# 🤖 4️⃣ Model Training

### Models Tested

| Model             | R² Score | RMSE       |
| ----------------- | -------- | ---------- |
| Linear Regression | 0.62     | Medium     |
| Random Forest     | 0.83     | Low        |
| XGBoost           | **0.88** | **Lowest** |

Best model: **XGBoost Regressor**

### Training Strategy

* Train/Test Split (80/20)
* 5-Fold Cross Validation
* Hyperparameter tuning via GridSearchCV
* Early stopping (for boosting models)

# 📊 5️⃣ Model Evaluation

Metrics Used:

* R² Score
* RMSE
* MAE

Error Analysis:

* Slight underprediction for premium airlines
* Higher variance for rare routes
* Model generalizes well across most routes

# 🚀 6️⃣ Deployment-Ready Structure

Project organized for scalability:

```bash
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
├── models/
├── api/
│   └── app.py
└── README.md
```

# 🌐 7️⃣ Optional Production Extension

### Can Be Extended To:

* REST API (FastAPI)
* Dockerized deployment
* Streamlit dashboard
* CI/CD integration
* MLflow experiment tracking
* Model monitoring

# 🛠 Tech Stack

* Python
* Pandas
* NumPy
* Seaborn / Matplotlib
* Scikit-learn
* XGBoost
