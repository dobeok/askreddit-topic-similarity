from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
pd.set_option('display.max_colwidth', 500)

np.random.seed(9) # to fix sampled data


orig_corpus = []
with open('data/1000posts.txt') as file:
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
right_table = pd.DataFrame(corpus, columns=['thread2'])

count_result = count_result.merge(right_table, left_on='best_match', right_index=True)


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
right_table = pd.DataFrame(corpus, columns=['thread2'])


tfidf_result = tfidf_result.merge(right_table, left_on='best_match', right_index=True)


# compare
# random 10
sample_index = np.random.randint(0, len(corpus), 10)

count_result.loc[sample_index][['thread1', 'thread2', 'similarity']].to_markdown(index=False)
tfidf_result.loc[sample_index][['thread1', 'thread2', 'similarity']].to_markdown(index=False)


bins = [_/10 for _ in range(0, 11)]

fig, ax = plt.subplots()
ax.clear()
pd.cut(tfidf_result['similarity'], bins=bins).value_counts().sort_index().plot(ax=ax, kind='bar', width=1, ec='white', alpha=.3, color='#fb553b', label='tf-idf')
pd.cut(count_result['similarity'], bins=bins).value_counts().sort_index().plot(ax=ax, kind='bar', width=1, ec='white', alpha=.3, label='raw count')


ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=2)



fig.savefig('compare sim score.png', bbox_inches='tight')