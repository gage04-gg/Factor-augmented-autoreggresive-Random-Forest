# Factor-Augmented Autoregressive Random Forest

This repository contains a Python implementation of a **Factor-Augmented Autoregressive Random Forest (FAARF)** framework for forecasting U.S. macroeconomic time series using the **FRED-QD** dataset.

The project combines autoregressive information with latent factors extracted from a large macroeconomic dataset and employs Random Forest models to capture nonlinear relationships for improved forecasting performance.

---

## Overview

Macroeconomic forecasting involves predicting economic indicators using historical observations and a large number of correlated variables. Traditional linear models often struggle to capture nonlinear interactions present in macroeconomic data.

This project implements a **Factor-Augmented Autoregressive Random Forest** approach by combining:

- Autoregressive lag features
- Latent macroeconomic factors
- Random Forest regression

to forecast macroeconomic variables and evaluate predictive performance.

---

## My Contribution

This repository contains my implementation of the forecasting pipeline, including:

- Data preprocessing
- FRED-QD dataset handling
- Feature engineering
- Factor extraction
- Autoregressive feature construction
- Random Forest model training
- Forecast generation
- Model evaluation and visualization

This project is intended for research and educational purposes.

---

## Features

- Factor-Augmented Random Forest forecasting
- Autoregressive feature generation
- Principal Component Analysis (PCA) for factor extraction
- Expanding/Rolling window forecasting
- Hyperparameter tuning
- Performance evaluation
- Forecast visualization

---

## Dataset

Experiments are performed using the **FRED-QD (Federal Reserve Economic Data - Quarterly Database)**.

The dataset contains hundreds of U.S. macroeconomic indicators including:

- GDP
- Inflation
- Employment
- Industrial Production
- Interest Rates
- Housing
- Financial Variables

The dataset is not included in this repository. Please download it from the Federal Reserve Bank of St. Louis.

---

## Project Structure

```
Factor-augmented-autoregressive-Random-Forest/
│
├── data/
├── preprocessing/
├── models/
├── forecasting/
├── evaluation/
├── notebooks/
├── results/
├── figures/
├── requirements.txt
├── main.py
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/gage04-gg/Factor-augmented-autoreggresive-Random-Forest.git
cd Factor-augmented-autoreggresive-Random-Forest
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the complete forecasting pipeline

```bash
python main.py
```

---

## Evaluation Metrics

Model performance is evaluated using:

- RMSE
- MAE
- Mean Squared Error (MSE)
- R² Score

---

## Results

The implemented model forecasts macroeconomic variables using nonlinear Random Forest regression while incorporating latent economic factors and autoregressive dynamics.

Performance can be compared against conventional forecasting models such as:

- Autoregressive (AR)
- Linear Regression
- Random Forest
- Factor-Augmented Models

---

## Requirements

- Python 3.10+
- NumPy
- pandas
- scikit-learn
- matplotlib
- scipy

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

## References

If this repository contributes to your research, please cite the original work on macroeconomic random forests and the FRED-QD dataset.

- Goulet Coulombe, P. (2024). *The Macroeconomy as a Random Forest*. Journal of Applied Econometrics.
- McCracken, M. W., & Ng, S. (2016). *FRED-QD: A Quarterly Database for Macroeconomic Research.*

---

## Disclaimer

This repository is an independent implementation of a factor-augmented autoregressive random forest forecasting framework for research and educational purposes. Credit for the original methodologies belongs to their respective authors.
