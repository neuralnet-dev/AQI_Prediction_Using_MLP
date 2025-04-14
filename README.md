
# AQI Prediction Using MLP Regressor

This project applies a **Multilayer Perceptron (MLP) Regressor** to predict **Air Quality Index (AQI)** based on pollutant data collected from Indian cities. The model is trained using the `city_day.csv` dataset, which contains daily air pollution readings along with AQI values.
Built and developed by Amith S Patil, Asher Jarvis Pinto, Henry Gladson, Fariza Nuha Farooq and Lavanya Devadiga.

---

## Dataset

- **Source**: [city_day.csv](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
- **Features**: Pollutants such as PM2.5, PM10, NO2, SO2, CO, O3, etc.
- **Target**: `AQI` (Air Quality Index)

---

## Workflow

1. **Data Preprocessing**
   - Dropped missing values in the `AQI` column
   - Filled other missing entries with **median values**
   - Encoded `City` column using **Label Encoding**

2. **Exploratory Data Analysis**
   - Visualized the correlation between features using a **heatmap**

3. **Modeling**
   - Applied **StandardScaler** to normalize features
   - Built an MLP Regressor with:
     - Hidden layers: `(64, 32)`
     - Activation: `'logistic'`
     - Optimizer: `'adam'`
     - Iterations: `1000`

4. **Evaluation**
   - Metrics:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - R² Score
   - Visuals:
     - **Scatter plot** of predicted vs. actual AQI
     - **Line plot** comparing actual and predicted AQI over a sample subset

---

## Model Performance ( Logistic Activation )

```
Mean Absolute Error: 0.16124993656149952
Mean Squared Error: 0.10658939570522047
R-squared (R²) Score: 0.8883794196602596
```

Make sure `city_day.csv` is in the same directory.
