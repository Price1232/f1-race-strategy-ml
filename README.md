# f1-race-strategy-ml
Machine learning based simulation of Formula 1 race strategy using historical race data.

# F1 Race Strategy ML

Machine Learning based simulation of Formula 1 race strategy using historical race data.

This project predicts lap times using machine learning and simulates race strategies to evaluate optimal pit stop timing and tyre usage.

---

## Overview

Race strategy is one of the most important aspects of Formula 1. Teams must decide:

- when to pit
- which tyre compound to use
- how tyre degradation affects lap times
- how safety cars and traffic influence strategy

This project builds a **machine learning based simulator** to analyze these decisions using historical race data.

---

## Features

### Lap Time Prediction

A machine learning model predicts lap times based on:

- tyre life
- tyre compound
- fuel load
- track temperature
- air temperature
- circuit characteristics
- driver
- team

---

### Strategy Simulation

The strategy engine simulates different pit strategies including:

- 0 stop
- 1 stop
- 2 stop
- early or late pit windows

Each strategy is evaluated based on predicted total race time.

---

### Monte Carlo Simulation

Monte Carlo simulations are used to evaluate uncertainty in race outcomes.

The simulation includes:

- lap time randomness
- safety car probability
- pit stop variability
- traffic effects

This allows estimation of:

- strategy win probability
- distribution of race times
- robustness of strategies

---

### Race Replay

The system can replay historical races and compare:

- actual race strategies
- machine learning predicted strategies

This allows evaluation of model accuracy.

---

## System Architecture
FastF1 API
    │
    ▼
Data Processing
    │
    ▼
Feature Engineering
    │
    ▼
ML Lap Time Model
    │
    ▼
Strategy Simulation Engine
    │
    ▼
Monte Carlo Simulation
    │
    ▼
Streamlit Dashboard

---

## Technologies Used

- Python
- FastF1
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Plotly

---

## Machine Learning Model

The lap time prediction model uses a **Scikit-Learn pipeline** with:

- StandardScaler for numeric features
- OneHotEncoder for categorical features
- GradientBoostingRegressor

The model predicts lap time using multiple race and circuit features.

---

The dashboard allows interactive exploration of race strategies and simulations.

---

## Example Analysis

The simulator can be used to analyze:

- optimal pit windows
- tyre degradation effects
- safety car strategy adjustments
- race strategy comparisons

---
## Model Performance

The lap-time prediction model was trained using historical Formula 1 telemetry data.

Model: Gradient Boosting Regressor  
Training Data: FastF1 telemetry datasets  
Train/Test Split: 80/20  

Performance:

MAE: **1.54 seconds**

The model predicts lap times using features including:

- Tyre life
- Fuel load estimation
- Compound type
- Track temperature
- Driver and team encoding
- Circuit characteristics

## Future Work

Possible extensions include:

- reinforcement learning for strategy optimization
- real-time telemetry integration
- driver performance modeling
- cloud deployment for large-scale simulations
