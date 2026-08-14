# Car Price Prediction

A Machine Learning project that predicts used car prices based on different car features.

## Project Overview

This project uses Machine Learning regression algorithms to predict the price of a car.

The dataset contains features such as:

- Brand
- Model
- Model Year
- Mileage
- Fuel Type
- Transmission
- Exterior Color
- Interior Color
- Engine information
- Other car specifications

## Machine Learning

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Missing Value Handling
- Feature Engineering
- Categorical Encoding
- Model Training
- Hyperparameter Tuning using Optuna
- Model Evaluation
- Streamlit Deployment

## Model

XGBoost Regressor is used for car price prediction.

Hyperparameters are optimized using Optuna.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- Matplotlib
- Seaborn
- Streamlit

## Project Structure

```text
CarPricePrediction/
│
├── app.py
├── train.py
├── car_price_prediction_.csv
├── requirements.txt
├── README.md
└── .gitignore
