import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import difflib
from sklearn.metrics.pairwise import cosine_similarity
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

new_df = df['genres'].astype(str) + '' + df['keywords'].astype(str) + '' + df['original_langage'].astype(str) + '' + df['cast'].astype(str) + '' + df['director'].astype(str) + '' + df['release_date'].astype(str) + '' + df['runtime'].astype(str) + '' + df['revenue'].astype(str) + '' + df["homepage"].astype(str) + '' + df['tagline'].astype(str) + '' + df['title'].astype(str) + '' + df['vote_average'].astype(str) + '' + df['vote_count'].astype(str) + '' + df['overview'].astype(str)
new_df

"""**First five values of new data frame**"""

new_df.head()

"""**Checking for any null values in the new data frame**"""

new_df.isnull().sum()

df.shape

"""**Converting Text to numerical Value**"""

#converting text data to numberical values
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
cv_fit = cv.fit_transform(new_df)

print(cv_fit)

similarity= cosine_similarity(feat_vectors)
print(similarity)

similarity.shape

movie_name=input('enter your favourite movie name: ')

"""**Creating a list of movies given in the dataset**"""

#creating a list of movies given in the dataset
list_of_all_titles=df['title'].to_list()
print(list_of_all_titles)

"""**Find close match for movie name given by user**"""

find_close_match= difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)

