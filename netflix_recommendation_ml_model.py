# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

"""**Read file**"""

df = pd.read_csv("/content/movies (1).csv")

"""**Check the information of the dataset**"""

df.info()

"""**Check null values**"""

df.isnull().sum()

"""**Shape of the dataset**"""

df.shape

"""**Check top five values**"""

df.head()

"""**Find out the column name**"""

df.columns

"""**Selecting the appropiate columns for training model**"""

selected_cols=['genres','keywords','tagline','cast','director']
print(selected_cols)

"""**Replacing null values with null string**"""

for feat in selected_cols:
    df[feat]=df[feat].fillna('')

"""**Combining all selected feature**"""

new_df=df['genres']+''+df['keywords']+''+df['tagline']+''+df['cast']+''+df['director']
new_df

"""**First five values of new data frame**"""

new_df.head()

"""**Checking for any null values in the new data frame**"""

new_df.isnull().sum()

