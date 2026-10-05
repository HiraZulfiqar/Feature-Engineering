import pandas as pd

# Load dataset
df = pd.read_csv("datasets/employees.csv")

print("________ RAW DATA ________")
print(df)

print("\n________ DATA TYPES ________")
print(df.dtypes)

df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])
print(df['Joining_Date'])

df['Joining_Year'] = df['Joining_Date'].dt.year
df['Joining_Month'] = df['Joining_Date'].dt.month

print("\n________ DATE FEATURES ________")
print(df[['Joining_Date', 'Joining_Year', 'Joining_Month']])

df['Joining_Day'] = df['Joining_Date'].dt.day_name()

print("\n________ DAY FEATURE ________")
print(df[['Joining_Date', 'Joining_Day']])

print("\n________ CURRENT COLUMNS ________")
print(df.columns)

# ==========================================
# STEP 3 — CREATE DATE-BASED FEATURES
# ==========================================

# Create a new feature: how long the employee has been with the company
today = pd.Timestamp.today()

df['Years_With_Company'] = (
    (today - df['Joining_Date']).dt.days / 365.25
).round(1)

print("\n________ FEATURE CREATION ________")
print(df[['Joining_Date',
          'Joining_Year',
          'Joining_Month',
          'Joining_Day',
          'Years_With_Company']])

# ==========================================
# STEP 4 — CATEGORICAL ENCODING
# ==========================================

# Remove identifier column
df = df.drop(columns=['Employee_Code'])

# Remove original date because we already extracted useful features
df = df.drop(columns=['Joining_Date'])

# Find categorical columns
categorical_columns = df.select_dtypes(
    include=['object', 'category']
).columns

print("\n________ CATEGORICAL COLUMNS ________")
print(categorical_columns)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True
)

print("\n________ AFTER ENCODING ________")
print(df.head())

print("\n________ FINAL COLUMNS ________")
print(df.columns)

# ==========================================
# STEP 5 — FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

# Select numerical columns
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

print("\n________ NUMERICAL COLUMNS ________")
print(numerical_columns)

# Create scaler
scaler = StandardScaler()

# Scale numerical features
df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)

print("\n________ AFTER SCALING ________")
print(df.head())

print("\n________ NUMERICAL SUMMARY ________")
print(df[numerical_columns].describe())

# ==========================================
# STEP 5 — HANDLING MISSING VALUES
# ==========================================

print("\n________ MISSING VALUES BEFORE ________")
print(df.isnull().sum())

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

categorical_columns = df.select_dtypes(
    include=['object', 'category']
).columns

# Fill numerical missing values with median
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\n________ MISSING VALUES AFTER ________")
print(df.isnull().sum())

# ==========================================
# STEP 6 — FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

# Find numerical columns
numerical_columns = df.select_dtypes(
    include=['int64', 'float64']
).columns

print("\n________ NUMERICAL COLUMNS BEFORE SCALING ________")
print(df[numerical_columns])

# Create scaler
scaler = StandardScaler()

# Apply Standardization
df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)

print("\n________ NUMERICAL COLUMNS AFTER SCALING ________")
print(df[numerical_columns])

print("\n________ SUMMARY AFTER SCALING ________")
print(df[numerical_columns].describe())

# ==========================================
# STEP 7 — OUTLIER DETECTION & HANDLING
# ==========================================

import numpy as np

# Numerical columns
numerical_columns = ['Age', 'Income', 'Years_With_Company']

print("\n________ OUTLIERS ________")

for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(f"\n{column}")
    print("Lower Limit:", lower_limit)
    print("Upper Limit:", upper_limit)
    print("Number of Outliers:", len(outliers))

# ==========================================
# HANDLE OUTLIERS
# ==========================================

# Cap extreme values instead of deleting rows
for column in numerical_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    df[column] = np.clip(
        df[column],
        lower_limit,
        upper_limit
    )

print("\n________ AFTER OUTLIER HANDLING ________")
print(df[numerical_columns])

# ==========================================
# STEP 8 — FINAL ML-READY DATASET
# ==========================================

print("\n==========================================")
print("        FINAL ML-READY DATASET")
print("==========================================")

print("\n________ FINAL SHAPE ________")
print(df.shape)

print("\n________ FINAL COLUMNS ________")
print(df.columns)

print("\n________ FINAL DATA ________")
print(df)

print("\n________ FINAL DATA TYPES ________")
print(df.dtypes)

print("\n________ FINAL MISSING VALUES ________")
print(df.isnull().sum())