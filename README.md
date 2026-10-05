# Feature Engineering & Data Preprocessing

A practical Machine Learning data preprocessing project that transforms raw employee data into a clean and ML-ready dataset.

## What This Project Does

This project demonstrates important Feature Engineering and Data Preprocessing techniques:

* Converts dates into useful features
* Extracts joining year, month, and day
* Calculates years an employee has been with the company
* Removes unnecessary columns
* Performs One-Hot Encoding
* Handles missing values
* Applies Standard Scaling
* Detects outliers using the IQR method
* Handles outliers using value capping
* Produces a final ML-ready dataset

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn

## Project Structure

```text
Feature-Engineering/
│
├── datasets/
│   └── employees.csv
│
├── main.py
├── README.md
└── requirements.txt
```

## Feature Engineering Workflow

```text
Raw Employee Data
        ↓
Date Feature Extraction
        ↓
Feature Creation
        ↓
Remove Unnecessary Features
        ↓
One-Hot Encoding
        ↓
Missing Value Handling
        ↓
Feature Scaling
        ↓
Outlier Detection
        ↓
Outlier Handling
        ↓
ML-Ready Dataset
```

## Techniques Used

### 1. Date Feature Engineering

Extracts useful information from `Joining_Date`:

* Joining Year
* Joining Month
* Joining Day
* Years With Company

### 2. Categorical Encoding

Converts categorical columns into numerical values using **One-Hot Encoding**.

### 3. Missing Value Handling

* Numerical values → Median
* Categorical values → Mode

### 4. Feature Scaling

Uses `StandardScaler` to standardize numerical features.

### 5. Outlier Handling

Uses the **IQR (Interquartile Range)** method to detect outliers and caps extreme values instead of deleting rows.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

## Learning Goal

This project was created to practice the most important data preprocessing and feature engineering techniques used before feeding data into a Machine Learning model.
