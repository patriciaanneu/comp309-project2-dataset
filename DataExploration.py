# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:18:23 2026

@author: patri
"""

import pandas as pd

#load the data
file = "https://raw.githubusercontent.com/patriciaanneu/comp309-project2-dataset/refs/heads/main/HATE_CRIME_OPEN_DATA_-1768225117781365955.csv"
data_hate_crimes = pd.read_csv(file, sep=",")

#display the column name
print(data_hate_crimes.columns.values)
#display the shape of the data frame
print(data_hate_crimes.shape)
#display the main statistics of the data
print(data_hate_crimes.describe())
# display the types of columns
print(data_hate_crimes.dtypes)

#display information about the dataset columns
print("\nDataset Information:")
print(data_hate_crimes.info())

#display the number of unique values in each column
print("\nUnique Values:")
print(data_hate_crimes.nunique())

#statistical analysis of the dataset
print("\nStatistical Summary:")
print(data_hate_crimes.describe(include="all"))

#count the number of occurences by year
print("\nHate Crime Occurrences by Year:")
print(data_hate_crimes["OCCURRENCE_YEAR"].value_counts().sort_index())

#count the number of arrests made
print("\nArrest Distribution:")
print(data_hate_crimes["ARREST_MADE"].value_counts())

#calculate the percentage of arrests made
print("\nArrest Percentages:")
print(data_hate_crimes["ARREST_MADE"].value_counts(normalize=True) * 100)

#count the number of occurences by primary offence
print("\nOccurrences by Primary Offence:")
print(data_hate_crimes["PRIMARY_OFFENCE"].value_counts())

#check for missing values in each column
print("\nMissing Values:")
print(data_hate_crimes.isnull().sum())

#calculate the percentage of missing values
print("\nMissing Values Percentage:")
print((data_hate_crimes.isnull().sum() / len(data_hate_crimes)) * 100)

#calculate the total missing values
print("\nTotal Missing Values:")
print(data_hate_crimes.isnull().sum().sum())

import matplotlib.pyplot as plt

#count hate crime occurrences by year
occurrences_by_year = data_hate_crimes["OCCURRENCE_YEAR"].value_counts().sort_index()

#create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(occurrences_by_year.index.astype(str), occurrences_by_year.values)
plt.title("Hate Crime Occurrences by Year")
plt.xlabel("Occurrence Year")
plt.ylabel("Number of Occurrences")

plt.tight_layout()
plt.show()

#count hate crime occurrences by police division
occurrences_by_division = data_hate_crimes["DIVISION"].value_counts().sort_index()

#create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(occurrences_by_division.index.astype(str), occurrences_by_division.values)
plt.title("Hate Crime Occurrences by Police Division")
plt.xlabel("Police Division")
plt.ylabel("Number of Occurrences")

plt.tight_layout()
plt.show()


#count hate crime occurrences by primary offence
occurrences_by_offence = data_hate_crimes["PRIMARY_OFFENCE"].value_counts()

#create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(occurrences_by_offence.index, occurrences_by_offence.values)

plt.title("Hate Crime Occurrences by Primary Offence")
plt.xlabel("Primary Offence")
plt.ylabel("Number of Occurrences")

#rotate labels for better readability
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()

#count hate crime occurrences by arrest status
arrest_distribution = data_hate_crimes["ARREST_MADE"].value_counts()

#create a bar chart
plt.figure(figsize=(7, 5))
plt.bar(arrest_distribution.index.astype(str), arrest_distribution.values)

plt.title("Hate Crime Occurrences by Arrest Status")
plt.xlabel("Arrest Made")
plt.ylabel("Number of Occurrences")

plt.tight_layout()
plt.show()


# Count occurrences for each bias category

#yes/no bias columns
yes_no_columns = [
    "AGE_BIAS",
    "MENTAL_OR_PHYSICAL_DISABILITY",
    "MULTIPLE_BIAS"
]

#other bias columns
other_bias_columns = [
    "RACE_BIAS",
    "ETHNICITY_BIAS",
    "LANGUAGE_BIAS",
    "RELIGION_BIAS",
    "SEXUAL_ORIENTATION_BIAS",
    "GENDER_BIAS"
]

#count yes values for yes/no columns
yes_counts = (
    data_hate_crimes[yes_no_columns]
    .apply(lambda column: column.str.strip().str.lower().eq("yes"))
    .sum()
)

#count recorded entries for other bias columns
other_counts = data_hate_crimes[other_bias_columns].count()

#combine the counts
bias_counts = pd.concat([yes_counts, other_counts])

print("\nHate Crime Occurrences by Bias Category:")
print(bias_counts)

#count missing values for each bias category
bias_missing = data_hate_crimes[
    yes_no_columns + other_bias_columns
].isnull().sum()

print("\nMissing Values by Bias Category:")
print(bias_missing)

#create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(bias_counts.index, bias_counts.values)

plt.title("Hate Crime Occurrences by Bias Category")
plt.xlabel("Bias Category")
plt.ylabel("Number of Recorded Occurrences")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()


#count hate crime occurrences by location type
location_counts = data_hate_crimes["LOCATION_TYPE"].value_counts()

#display the counts
print("\nHate Crime Occurrences by Location Type:")
print(location_counts)

#create a bar chart
plt.figure(figsize=(12, 6))
plt.bar(location_counts.index, location_counts.values)
plt.title("Hate Crime Occurrences by Location Type")
plt.xlabel("Location Type")
plt.ylabel("Number of Occurrences")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()


