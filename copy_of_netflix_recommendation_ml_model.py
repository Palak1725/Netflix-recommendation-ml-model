# -*- coding: utf-8 -*-


import pandas as pd

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
