from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

content_features = ["genres", "cast", "keywords", "original_language", "overview", "spoken_language", "production_countries", "runtime"]

collab_features = ["homepage", "tagline", "title", "vote_average", "vote_count"]

metadata_features = ["release_data", "runtime", "revenue"]

content_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Converts movie descriptions into TF-IDF features.
    ('scaler', StandardScaler(with_mean=False))
])

collab_pipeline = Pipeline([
    ('scaler', StandardScaler())
])

metadata_pipeline = Pipeline([
    ('encoder', OneHotEncoder()),  # Converts categorical metadata like genres into numbers.
    ('scaler', StandardScaler(with_mean=False))
])

combined_features = FeatureUnion([
    ("content", content_pipeline),
    ("collaborative", collab_pipeline),
    ("metadata", metadata_pipeline)
])

combined_data = combined_features.fit_transform(df)
