# Importing Dependencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

df = pd.read_csv("city_day.csv")

data_info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "head": df.head(),
    "info": df.info(),
    "missing_values": df.isnull().sum()
}

print("Data Information:")
print("-" * 30)

print(f"Shape: {data_info['shape']}")
print("\nColumns:")
for col in data_info['columns']:
        print(f"  - {col}")

print("\nInfo:")
print(data_info['info'])

print("\nMissing Values:")
print(data_info['missing_values'])

print("\nHead:")
print(data_info['head'])

df = df.dropna(subset="AQI")
df = df.drop(columns=["Date", "AQI_Bucket"], axis=1)
for col in df.columns:
    if df[col].isnull().sum()>0:
        df[col]=df[col].fillna(df[col].median())
df.isnull().sum()


#using LabelEncoder to use City: a str data column 
df1 = df['City'].unique()
print("\nUnique City Names:")
for name in df1:
        print(f"  - {name}") #lists out all city names
label_encoder = LabelEncoder()
df['City'] = LabelEncoder().fit_transform(df['City']) 

# Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Pollutants and AQI")
plt.show()

# Model Training
x = df.drop(columns=["AQI"])
y = df["AQI"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_train = scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test = scaler.transform(y_test.values.reshape(-1, 1))

mlp = MLPRegressor(hidden_layer_sizes=(64, 32),
                   activation='relu', solver='adam',
                    max_iter=1000, random_state=42)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(x_test)

# Predictions on Training Set
y_train_pred = mlp.predict(x_train)

# Training Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

print("\nTraining Performance:")
print("-" * 30)
print("Mean Absolute Error (Train):", train_mae)
print("Mean Squared Error (Train):", train_mse)
print("R-squared (Train):", train_r2)




# Evaluation metrics - MAE, MSE and R^2
mae = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
r2 = r2_score(y_test, y_pred)
print("R-squared (R²) Score:", r2) 
plt.figure(figsize=(8, 6))

# Result Plot
plt.scatter(y_test, y_pred, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Scatter Plot: Predicted vs. Actual Performance Index", fontsize=14)
plt.xlabel("Actual Performance Index", fontsize=12)
plt.ylabel("Predicted Performance Index", fontsize=12)
plt.grid(alpha=0.3)
plt.show()

subset_size = 100 
points = np.arange(subset_size)
y_test_subset = y_test[:subset_size]
y_pred_subset = y_pred[:subset_size]
plt.figure(figsize=(12, 6))  # Adjust figure size
plt.plot(points, y_test_subset, color='red', label='Actual AQI')
plt.plot(points, y_pred_subset, color='green', label='Predicted AQI')
plt.title("Actual vs Predicted AQI (Subset)")
plt.xlabel("Data Points")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)
plt.show()

