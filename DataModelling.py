# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:58:43 2026

@author: patri
"""

import pandas as pd

#load the dataset
file = "https://raw.githubusercontent.com/patriciaanneu/comp309-project2-dataset/refs/heads/main/HATE_CRIME_OPEN_DATA_-1768225117781365955.csv"
data_hate_crimes = pd.read_csv(file, sep=",")

#check for duplicate records
duplicate_count = data_hate_crimes.duplicated().sum()

print("\nNumber of Duplicate Records:")
print(duplicate_count)

#display rows where LOCATION_TYPE is missing
missing_locations = data_hate_crimes[
    data_hate_crimes["LOCATION_TYPE"].isnull()
]

print(missing_locations[["EVENT_UNIQUE_ID", "LOCATION_TYPE"]])

#replace missing location types with Unknown
data_hate_crimes["LOCATION_TYPE"] = data_hate_crimes["LOCATION_TYPE"].fillna("Unknown")

#check missing values again
print("\nMissing LOCATION_TYPE Values:")
print(data_hate_crimes["LOCATION_TYPE"].isnull().sum())

#display updated location type counts
print("\nUpdated Location Type Counts:")
print(data_hate_crimes["LOCATION_TYPE"].value_counts())


#bias columns with missing values
bias_columns = [
    "RACE_BIAS",
    "ETHNICITY_BIAS",
    "LANGUAGE_BIAS",
    "RELIGION_BIAS",
    "SEXUAL_ORIENTATION_BIAS",
    "GENDER_BIAS"
]

#display the recorded values in each bias column
for column in bias_columns:
    print("\n" + column)
    print(data_hate_crimes[column].value_counts(dropna=False))
    

#replace missing bias values with Unknown
data_hate_crimes[bias_columns] = data_hate_crimes[bias_columns].fillna("Unknown")

#check missing values in bias columns
print("\nMissing Bias Values After Cleaning:")
print(data_hate_crimes[bias_columns].isnull().sum())


#check missing values across the entire dataset
print("\nMissing Values After Cleaning:")
print(data_hate_crimes.isnull().sum())

#display the total number of missing values
print("\nTotal Missing Values:")
print(data_hate_crimes.isnull().sum().sum())


#display data types of all columns
print("\nDataset Data Types:")
print(data_hate_crimes.dtypes)


#convert date columns to datetime format
data_hate_crimes["OCCURRENCE_DATE"] = pd.to_datetime(
    data_hate_crimes["OCCURRENCE_DATE"], errors="coerce"
)

data_hate_crimes["REPORTED_DATE"] = pd.to_datetime(
    data_hate_crimes["REPORTED_DATE"], errors="coerce"
)

#display updated date column types
print("\nUpdated Date Column Types:")
print(data_hate_crimes[["OCCURRENCE_DATE", "REPORTED_DATE"]].dtypes)

#check for invalid dates
print("\nInvalid Date Values:")
print(data_hate_crimes[["OCCURRENCE_DATE", "REPORTED_DATE"]].isnull().sum())


#standardize yes/no values
yes_no_columns = [
    "AGE_BIAS",
    "MENTAL_OR_PHYSICAL_DISABILITY",
    "MULTIPLE_BIAS",
    "ARREST_MADE"
]

for column in yes_no_columns:
    data_hate_crimes[column] = (
        data_hate_crimes[column]
        .str.strip()
        .str.lower()
        .str.capitalize()
    )

#standardize sexual orientation values
data_hate_crimes["SEXUAL_ORIENTATION_BIAS"] = (
    data_hate_crimes["SEXUAL_ORIENTATION_BIAS"]
    .replace({
        "2Slgbtq+": "2SLGBTQ+"
    })
)

#display standardized values
for column in yes_no_columns:
    print("\n" + column)
    print(data_hate_crimes[column].value_counts())

print("\nSEXUAL_ORIENTATION_BIAS")
print(data_hate_crimes["SEXUAL_ORIENTATION_BIAS"].value_counts())


#count unique values in each column
unique_counts = data_hate_crimes.nunique()

print("\nUnique Values in Each Column:")
print(unique_counts)

#identify columns with only one unique value
constant_columns = unique_counts[unique_counts == 1].index

print("\nColumns with Only One Unique Value:")
print(constant_columns.tolist())

#remove columns with only one unique value
data_hate_crimes = data_hate_crimes.drop(columns=constant_columns)

#display remaining columns
print("\nRemaining Dataset Columns:")
print(data_hate_crimes.columns.tolist())

#display updated dataset shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

#remove identifier columns
identifier_columns = [
    "OBJECTID",
    "EVENT_UNIQUE_ID"
]

data_hate_crimes = data_hate_crimes.drop(columns=identifier_columns)

#display updated dataset shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

#display remaining columns
print("\nRemaining Dataset Columns:")
print(data_hate_crimes.columns.tolist())

#remove redundant neighbourhood code columns
redundant_columns = [
    "HOOD_158",
    "HOOD_140"
]

data_hate_crimes = data_hate_crimes.drop(columns=redundant_columns)

#display updated dataset shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

#display remaining columns
print("\nRemaining Dataset Columns:")
print(data_hate_crimes.columns.tolist())

#display the data types of the remaining columns
print("\nUpdated Dataset Data Types:")
print(data_hate_crimes.dtypes)

#remove redundant year columns
redundant_columns = [
    "OCCURRENCE_YEAR",
    "REPORTED_YEAR"
]

data_hate_crimes = data_hate_crimes.drop(columns=redundant_columns)

#display updated columns and shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

print("\nRemaining Dataset Columns:")
print(data_hate_crimes.columns.tolist())

#display the date ranges
print("\nOccurrence Date Range:")
print("Earliest:", data_hate_crimes["OCCURRENCE_DATE"].min())
print("Latest:", data_hate_crimes["OCCURRENCE_DATE"].max())

print("\nReported Date Range:")
print("Earliest:", data_hate_crimes["REPORTED_DATE"].min())
print("Latest:", data_hate_crimes["REPORTED_DATE"].max())

#calculate the number of days between occurrence and reporting
data_hate_crimes["REPORTING_DELAY"] = (
    data_hate_crimes["REPORTED_DATE"] -
    data_hate_crimes["OCCURRENCE_DATE"]
).dt.days

#display reporting delay statistics
print("\nReporting Delay Statistics:")
print(data_hate_crimes["REPORTING_DELAY"].describe())

#display the 10 records with the longest reporting delays
print("\nTop 10 Longest Reporting Delays:")

print(
    data_hate_crimes[
        ["OCCURRENCE_DATE", "REPORTED_DATE", "REPORTING_DELAY"]
    ]
    .sort_values(by="REPORTING_DELAY", ascending=False)
    .head(10)
)

#remove the reporting delay column
data_hate_crimes = data_hate_crimes.drop(columns=["REPORTING_DELAY"])

#display updated dataset shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

#display sample values from the time columns
print("\nOccurrence Time Samples:")
print(data_hate_crimes["OCCURRENCE_TIME"].head(10))

print("\nReported Time Samples:")
print(data_hate_crimes["REPORTED_TIME"].head(10))

#convert HHMM time into minutes since midnight
time_columns = ["OCCURRENCE_TIME", "REPORTED_TIME"]

for column in time_columns:
    hours = data_hate_crimes[column] // 100
    minutes = data_hate_crimes[column] % 100

    data_hate_crimes[column] = (hours * 60) + minutes

#display converted time values
print("\nConverted Time Samples:")
print(data_hate_crimes[time_columns].head(10))

#display minimum and maximum time values
print("\nOccurrence Time Range:")
print(data_hate_crimes["OCCURRENCE_TIME"].min())
print(data_hate_crimes["OCCURRENCE_TIME"].max())

print("\nReported Time Range:")
print(data_hate_crimes["REPORTED_TIME"].min())
print(data_hate_crimes["REPORTED_TIME"].max())

#extract month and day of the week from occurrence date
data_hate_crimes["OCCURRENCE_MONTH"] = (
    data_hate_crimes["OCCURRENCE_DATE"].dt.month
)

data_hate_crimes["OCCURRENCE_DAY"] = (
    data_hate_crimes["OCCURRENCE_DATE"].dt.day_of_week
)

#extract month and day of the week from reported date
data_hate_crimes["REPORTED_MONTH"] = (
    data_hate_crimes["REPORTED_DATE"].dt.month
)

data_hate_crimes["REPORTED_DAY"] = (
    data_hate_crimes["REPORTED_DATE"].dt.day_of_week
)

#display the extracted date features
print("\nExtracted Date Features:")
print(
    data_hate_crimes[
        ["OCCURRENCE_MONTH", "OCCURRENCE_DAY",
         "REPORTED_MONTH", "REPORTED_DAY"]
    ].head(10)
)

#remove original date columns
date_columns = ["OCCURRENCE_DATE", "REPORTED_DATE"]

data_hate_crimes = data_hate_crimes.drop(columns=date_columns)

#display updated dataset shape
print("\nUpdated Dataset Shape:")
print(data_hate_crimes.shape)

print("\nRemaining Dataset Columns:")
print(data_hate_crimes.columns.tolist())

#check for missing values in the updated dataset
print("\nMissing Values:")
print(data_hate_crimes.isnull().sum())

print("\nTotal Missing Values:")
print(data_hate_crimes.isnull().sum().sum())

# convert ARREST_MADE into numerical values
data_hate_crimes["ARREST_MADE"] = (
    data_hate_crimes["ARREST_MADE"].map({"No": 0, "Yes": 1})
)

#display target values
print("\nTarget Class Distribution:")
print(data_hate_crimes["ARREST_MADE"].value_counts())

#separate features (X) and target (y)
X = data_hate_crimes.drop(columns=["ARREST_MADE"])
y = data_hate_crimes["ARREST_MADE"]

#display the shapes
print("\nFeatures Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

#identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["str"]).columns.tolist()
numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

print("\nCategorical Features:")
print(categorical_columns)

print("\nNumerical Features:")
print(numerical_columns)

print("\nNumber of Categorical Features:")
print(len(categorical_columns))

print("\nNumber of Numerical Features:")
print(len(numerical_columns))

#display target class distribution
print("\nTarget Class Distribution:")
print(y.value_counts())

#display target class percentages
print("\nTarget Class Percentages:")
print(y.value_counts(normalize=True) * 100)


from sklearn.model_selection import train_test_split

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#display the dataset sizes
print("\nTraining Features Shape:")
print(X_train.shape)

print("\nTesting Features Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

from sklearn.preprocessing import OneHotEncoder

#create the OneHotEncoder
encoder = OneHotEncoder(handle_unknown="ignore")

#fit the encoder using the training data
X_train_encoded = encoder.fit_transform(X_train[categorical_columns])

#transform the testing data
X_test_encoded = encoder.transform(X_test[categorical_columns])

#display the encoded dataset shapes
print("\nEncoded Training Features Shape:")
print(X_train_encoded.shape)

print("\nEncoded Testing Features Shape:")
print(X_test_encoded.shape)

from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler

#scale numerical features
scaler = StandardScaler()

X_train_numerical = scaler.fit_transform(
    X_train[numerical_columns]
)

X_test_numerical = scaler.transform(
    X_test[numerical_columns]
)

# Combine categorical and numerical features
X_train_final = hstack([
    X_train_encoded,
    X_train_numerical
])

X_test_final = hstack([
    X_test_encoded,
    X_test_numerical
])

#display final feature shapes
print("\nFinal Training Features Shape:")
print(X_train_final.shape)

print("\nFinal Testing Features Shape:")
print(X_test_final.shape)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#create the Logistic Regression model
logistic_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

#create the RFE feature selector
rfe = RFE(
    estimator=logistic_model,
    n_features_to_select=12
)

#perform feature selection using training data
X_train_selected = rfe.fit_transform(X_train_final, y_train)

#apply the selected features to testing data
X_test_selected = rfe.transform(X_test_final)

#display selected feature shapes
print("\nSelected Training Features Shape:")
print(X_train_selected.shape)

print("\nSelected Testing Features Shape:")
print(X_test_selected.shape)

#get the selected feature names
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

all_feature_names = list(encoded_feature_names) + numerical_columns

selected_features = [
    all_feature_names[i]
    for i in range(len(all_feature_names))
    if rfe.support_[i]
]

#display selected features
print("\nSelected Features:")
for feature in selected_features:
    print(feature)

print("\nNumber of Selected Features:")
print(len(selected_features))

#train the Logistic Regression model
logistic_model.fit(X_train_selected, y_train)

#predict arrest outcomes using testing data
y_pred = logistic_model.predict(X_test_selected)

#display predictions
print("\nLogistic Regression Predictions:")
print(y_pred[:20])

from sklearn.metrics import accuracy_score

#calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

#display model accuracy
print("\nLogistic Regression Accuracy:")
print(accuracy)

print("\nLogistic Regression Accuracy Percentage:")
print(accuracy * 100, "%")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

#display confusion matrix
print("\nLogistic Regression Confusion Matrix:")
print(cm)

#visualize confusion matrix
cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Arrest", "Arrest Made"]
)

cm_display.plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

from sklearn.metrics import classification_report

#display classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Arrest", "Arrest Made"]
))

from sklearn.metrics import roc_curve, roc_auc_score

#get prediction probabilities
y_prob = logistic_model.predict_proba(X_test_selected)[:, 1]

#calculate ROC values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

#calculate AUC
auc_score = roc_auc_score(y_test, y_prob)

#display AUC
print("\nLogistic Regression AUC:")
print(auc_score)

#plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.show()