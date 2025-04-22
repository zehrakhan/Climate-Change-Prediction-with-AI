# Climate Change Prediction Using AI Models

This repository contains the code and documentation for my master's thesis, which explores long-term climate trends in Aachen, Germany using AI and time series modeling. The study is based on the Berkeley Earth dataset and focuses on identifying whether the climate is warming or cooling over time.

## 📘 Thesis Title
**"Development of an AI based prediction model for Detecting Climate Change through Long-Term Temperature Trend Analysis"**

## 📂 Structure of the Repository

## 📊 Dataset

- **Source**: Berkeley Earth
- **Time Range**: 1744 to 2013
- **Cities**: 81 German cities
- **Features**: AverageTemperature, TemperatureAnomaly, Latitude, Longitude, etc.

## 🧠 Models Implemented

The study compares traditional, machine learning, deep learning, and hybrid models:

### ✅ Traditional Time Series Models
- ARIMA  

### ✅ Machine Learning Models
- Random Forest Regressor  

### ✅ Deep Learning Models
- LSTM
- xLSTMTime
- Transformer

### ✅ Hybrid Approaches
- LSTM + Transformer  

## 🧪 Evaluation Strategy

- **Train/Validation/Test Split**: 70/15/15  
- **Feature**: `YearsSince1744`  
- **Target**: `AverageTemperature`  
- **Metrics**: MAE, RMSE, R² Score  
- **Loss Curves**: Included in `results/` folder  
- **Reproducibility**: `random_state=42` used throughout

## 🔍 Objective

The core objective is to model historical temperature trends and determine whether there is a significant warming or cooling pattern using different AI techniques.

## 📄 Thesis Report

- The thesis report is written in LaTeX.
- Source code is available in the zip folder: `Zahra_Thesis_Final_report_v9.zip`.
- The final PDF report can be found here: `Zahra_Thesis_Final_report_v9.pdf`.

## ⚙️ Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

