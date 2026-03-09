# f1-race-strategy-ml
Machine learning based simulation of Formula 1 race strategy using historical race data.

# Machine Learning Based Formula 1 Race Strategy Simulation

This project develops a machine learning based framework for predicting lap times and simulating race strategies in Formula 1 using historical race data.

## Motivation

Race strategy is a critical decision making component in motorsport. This project models tyre degradation and race dynamics using machine learning and simulation.

## System Overview

The system consists of three components:

1. Data ingestion from FastF1
2. Machine learning model for lap time prediction
3. Strategy simulation engine

## Machine Learning Model

A Gradient Boosting Regressor is trained to predict lap time based on:

- tyre life
- compound type
- fuel load
- track temperature
- air temperature
- circuit characteristics
- driver
- team

## Strategy Simulation

The simulator models:

- tyre degradation
- pit stop timing
- safety car scenarios
- traffic effects
- Monte Carlo simulations

## Dashboard

A Streamlit dashboard visualizes:

- strategy comparisons
- tyre degradation
- race replay
- lap time analysis

## Running the Project

Install dependencies:
