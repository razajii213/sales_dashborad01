# sales_dashborad01
Sales Analysis Dashboard 🛒 This project is a Sales Analysis Dashboard built using Python, Pandas, Matplotlib, and Seaborn. It analyzes sales data to extract useful insights and visualize them through graphs.

import pandas as pd
# Loading csv file into pandas dataframe
data = pd.read_csv('sales_data.csv')

# Previewing the first 5 rows of the dataframe
print(data.head())  # showing starting 5 rows
print("\nTotal Rows:", len(data))  # Total rows

# Converting 'Date' Column to DateTime Format with Error Handling
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Check for any 'NaT' (Invalid Dates)
invalid_dates = data['Date'].isna().sum()
print(f"\n 'Date' Column Converted Successfully! Invalid Dates Found: {invalid_dates}")

# if invalid dates found, print them
if invalid_dates > 0:
    print("\n Invalid Date Rows:")
    print(data[data['Date'].isna()])

# Data Types Check Again
print("\n Updated Data Types:")
print(data.dtypes)

# 'Sales' To create colums (Quantity × Price)
data['Sales'] = data['Quantity'] * data['Price']

# check for true or false
print(data.head())

import random

# Randomly 'Region' column addition
regions = ['North', 'South', 'East', 'West']
data['Region'] = [random.choice(regions) for _ in range(len(data))]

print(data.head())  # Check addition is seccessful or not

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  CSV Load kar rahe hain (Apna CSV path daal yahan)
data = pd.read_csv("sales_data.csv")

#  Columns ke spaces hata rahe hain (agar koi ho to)
data.columns = data.columns.str.strip()

#  'Sales' column create kar rahe hain (Quantity * Price)
if 'Sales' not in data.columns:
    data['Sales'] = data['Quantity'] * data['Price']  #  Yahan add kiya maine 'Sales' column
    print(" 'Sales' column create created!")
else:
    print(" 'Sales' column already have !")

#  Check kar rahe hain ki 'Sales' column hai ya nahi
if 'Sales' not in data.columns:
    print(" 'Sales' column not found. This are columns:")
    print(data.columns)
else:
    print("'Sales' column yes!")

# Setting style for graphs
sns.set(style='whitegrid')

# 1️ Sales Trend Over Time
plt.figure(figsize=(10, 6))
data.groupby('Date')['Sales'].sum().plot(marker='o', color='skyblue')
plt.title(' Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# 2️ Top 5 Products by Sales
top_products = data.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('🏆 Top 5 Products by Sales')
plt.xlabel('Total Sales')
plt.show()

# 3️ Sales by Category
plt.figure(figsize=(8, 5))
sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
plt.title(' Sales Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Sales')
plt.show()

# 4️ Region-wise Sales Distribution
if 'Region' in data.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Region', data=data, palette='coolwarm')
    plt.title(' Region-wise Sales Distribution')
    plt.xlabel('Region')
    plt.ylabel('Count of Sales')
    plt.show()
else:
    print(" 'Region' column nahi mila!")

# 5️⃣ Correlation Heatmap (Only for Numeric Data)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title(' Correlation Heatmap')
plt.show()

sns.countplot(x='Region', data=data, palette='coolwarm')
plt.title('🌍 Region-wise Sales Distribution')
plt.xlabel('Region')
plt.show()

print(data.dtypes)

# Sirf numeric columns ko extract kar rahe hain
numeric_data = data.select_dtypes(include=['int64', 'float64'])

print(numeric_data.head())  # Check karo sahi hai ki nahi
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title('📈 Correlation Heatmap')
plt.show()

print("Correlation between Price and Sales:", numeric_data['Price'].corr(numeric_data['Sales']))
print("Correlation between Quantity and Sales:", numeric_data['Quantity'].corr(numeric_data['Sales']))

plt.figure(figsize=(8, 5))
data.groupby('Category')['Sales'].sum().plot(kind='bar', color='coral')
plt.title('🛒 Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

plt.figure(figsize=(8, 5))
data.groupby('Region')['Sales'].sum().plot(kind='bar', color='teal')
plt.title('🌍 Total Sales by Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()

region_sales = data.groupby('Region')['Sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', colors=['#66b3ff', '#99ff99', '#ffcc99'])
plt.title('📊 Sales Distribution by Region')
plt.ylabel('')
plt.show()

data['Month'] = data['Date'].dt.to_period('M')

plt.figure(figsize=(10, 6))
data.groupby('Month')['Sales'].sum().plot(marker='o', linestyle='-', color='purple')
plt.title('📅 Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

top_products = data.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
print("Top 5 Best-Selling Products:\n", top_products)

plt.figure(figsize=(8, 5))
data.groupby('Category')['Sales'].sum().plot(kind='bar', color='orange')
plt.title('📦 Total Sales by Category')
plt.xlabel('Category')
plt.ylabel('Total Sales')
plt.show()

category_sales = data.groupby('Category')['Sales'].sum()
category_sales.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('📊 Sales Distribution by Category')
plt.ylabel('')
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Quantity', y='Sales', data=data, hue='Category', palette='viridis')
plt.title('📈 Sales vs. Quantity')
plt.xlabel('Quantity Sold')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

rint(" Conclusion:")
print("1. Laptop is the highest-selling product.")
print("2. Sales distribution across categories is balanced.")
print("3. Region-wise analysis shows which areas need improvement.")


# Extracting Month from 'Date' Column
data['Month'] = data['Date'].dt.month

# Monthly Sales Trend
plt.figure(figsize=(10, 6))
data.groupby('Month')['Sales'].sum().plot(marker='o', color='teal')
plt.title('📅 Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()

region_sales = data.groupby('Region')['Sales'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99', '#ffcc99'])
plt.title('🌍 Sales Contribution by Region')
plt.ylabel('')
plt.show()

# Sirf numeric columns ko extract kar rahe hain
numeric_data = data.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title('📈 Correlation Heatmap')
plt.show()

#  Check for missing values in 'Date' column
print(data['Date'].isnull().sum())

#  Remove rows with missing 'Date' values
data = data.dropna(subset=['Date'])

#  Converting 'Date' to numeric format for regression
data['DateNumeric'] = data['Date'].map(pd.Timestamp.toordinal)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)

 🏋️‍♂️ Training the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔮 Making Predictions
predictions = model.predict(X_test)

# 📊 Evaluating the Model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"📊 R² Score: {r2:.2f}")

 🔍 Feature Importance
importances = model.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.2f}")

import matplotlib.pyplot as plt

plt.barh(feature_names, importances, color='skyblue')
plt.title('📊 Feature Importance')
plt.xlabel('Importance')
plt.show()

rom sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(" Cross-Validation R² Scores:", scores)
print(" Average R² Score:", scores.mean())

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X, y)

print(" Best Parameters:", grid_search.best_params_)
print("📊 Best R² Score from Grid Search:", grid_search.best_score_)

# Train final model with best params
final_model = RandomForestRegressor(max_depth=None, min_samples_split=2, n_estimators=200)
final_model.fit(X, y)

# Predictions and performance
predictions = final_model.predict(X)
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f" Final Model Mean Squared Error: {mse:.2f}")
print(f" Final Model R² Score: {r2:.2f}")


import joblib
joblib.dump(final_model, 'final_sales_model.pkl')



