from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

cv = CountVectorizer()
cv_fit = cv.fit_transform(df)

similarity = cosine_similarity(cv_fit)
