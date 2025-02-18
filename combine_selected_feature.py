from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

content_features = ["genres", "cast", "keywords", "original_language", "overview", "spoken_language", "production_countries", "runtime"]

collab_features = ["homepage", "tagline", "title", "vote_average", "vote_count"]

metadata_features = ["release_data", "runtime", "revenue"]

new_df = df['genres'].astype(str) + '' + df['keywords'].astype(str) + '' + df['original_langage'].astype(str) + '' + df['cast'].astype(str) + '' + df['director'].astype(str) + '' + df['release_date'].astype(str) + '' + df['runtime'].astype(str) + '' + df['revenue'].astype(str) + '' + df["homepage"].astype(str) + '' + df['tagline'].astype(str) + '' + df['title'].astype(str) + '' + df['vote_average'].astype(str) + '' + df['vote_count'].astype(str) + '' + df['overview'].astype(str)

combined_data = combined_features.fit_transform(df)
