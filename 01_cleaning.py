import pandas as pd
import numpy as np

a = pd.read_csv("train.csv")  # read the Titanic dataset from the CSV file

print(a.head())  # show the first 5 rows
print(a.shape)  # show number of rows and columns
print(a.info())  # show column names, data types, and non-null counts
print(a.isnull().sum())    # isnull() → shows where missing values are
                            # sum() → counts them


def f(x):  # function to clean the dataset
    x["Age"] = pd.to_numeric(x["Age"], errors="coerce")  # make Age numeric, invalid values become NaN
    x["Fare"] = pd.to_numeric(x["Fare"], errors="coerce")  # make Fare numeric, invalid values become NaN

    b = x.isnull().sum()  # count missing values in each column and we make it again to verify it
    print(b)  # print missing values summary

    x["Age"] = x["Age"].fillna(x["Age"].median())  # fill missing Age values with the median
    x["Embarked"] = x["Embarked"].fillna(
        x["Embarked"].mode()[0])  # fill missing Embarked values with the most common value

    x = x.drop(columns=["Cabin"])  # remove Cabin because it has too many missing values

    c = x.duplicated().sum()  # count duplicate rows
    print(c)  # print number of duplicates
    x = x.drop_duplicates()  # remove duplicate rows

    d = x["Fare"].quantile(0.01)  # get the first quartile (0.01%)
    e = x["Fare"].quantile(0.99)  # get the third quartile (99%)
    g = e - d  # calculate IQR = Q3 - Q1
    h = d - 1.5 * g  # lower bound for outliers
    i = e + 1.5 * g  # upper bound for outliers

    j = ((x["Fare"] < h) | (x["Fare"] > i)).sum()  # count how many Fare values are outliers
    print(j)  # print number of outliers

    k = x["Fare"].quantile(0.99)  # get the 99th percentile of Fare
    x["Fare"] = x["Fare"].clip(upper=k)  # cap very large Fare values at the 99th percentile

    return x  # return the cleaned dataset


l = f(a)  # apply the cleaning function to the original dataset

print(l["Age"].isnull().sum())  # check if there are still missing values in Age
print((l["Fare"] > 0).all())  # check if all Fare values are greater than 0
print(l.shape)  # print final shape after cleaning

l.to_csv("cleaned_titanic.csv", index=False)  # save the cleaned dataset to a new CSV file








