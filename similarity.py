from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np


pd.set_option('display.max_colwidth', 500)

orig_corpus = []
with open('1000posts.txt') as file:
    for line in file:
        orig_corpus.append(line)


stemmer = SnowballStemmer('english')


def stem_tokens(text):
    text = text.strip()
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


corpus = [stem_tokens(_) for _ in orig_corpus]

# count
count_vectorizer = CountVectorizer(stop_words='english', min_df=0.005)
X_count = count_vectorizer.fit_transform(corpus)
df_count = pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names())

df_count_sim = pd.DataFrame(cosine_similarity(df_count, dense_output=True))
df_count_sim_as_np = df_count_sim.values

np.fill_diagonal(df_count_sim_as_np, 0)
df_count_result = pd.DataFrame(df_count_sim_as_np)
df_count_result['best_match'] = df_count_result.idxmax()
df_count_result['similarity'] = df_count_result.max()
count_result = df_count_result[['best_match', 'similarity']]
count_result['thread1'] = corpus
right_table = pd.DataFrame(corpus)

count_result = count_result.merge(right_table, left_on='best_match', right_index=True)
count_result.head()

# tf-idf
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names())

df_tfidf_sim = pd.DataFrame(cosine_similarity(df_tfidf, dense_output=True))
df_tfidf_sim_as_np = df_tfidf_sim.values

np.fill_diagonal(df_tfidf_sim_as_np, 0)
df_tfidf_result = pd.DataFrame(df_tfidf_sim_as_np)
df_tfidf_result['best_match'] = df_tfidf_result.idxmax()
df_tfidf_result['similarity'] = df_tfidf_result.max()

tfidf_result = df_tfidf_result[['best_match', 'similarity']]
tfidf_result['thread1'] = corpus
right_table = pd.DataFrame(corpus)

tfidf_result = tfidf_result.merge(right_table, left_on='best_match', right_index=True)
tfidf_result.head()
