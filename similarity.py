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
        orig_corpus.append(line.replace('\n', ''))


stemmer = SnowballStemmer('english')


def stem_tokens(text):
    text = text.strip()
    words = text.lower().split()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


corpus = [stem_tokens(_) for _ in orig_corpus]

# thread1: orignal list of thread name
# sim-tf: closest match, using term frequency
# sim-tf-idf: closest mathh, using term frequency - inverse document frequency
df = pd.DataFrame({
    'orig_thread': orig_corpus,
    'stemmed_thread': corpus})
df.head()


# using raw count
count_vectorizer = CountVectorizer(stop_words='english', min_df=0.005)
X_count = count_vectorizer.fit_transform(corpus)

# most frequent words
word_freq = pd.DataFrame({
    'word': count_vectorizer.get_feature_names_out(),
    'freq': X_count.toarray().sum(axis=0)
})
word_freq_15 = word_freq.sort_values(by='freq', ascending=False).head(15)

# plot
fig, ax = plt.subplots(figsize=(12, 3))
ax.bar(
    x=word_freq_15['word'].values,
    height=word_freq_15['freq'].values,
    width=1,
    ec='white'
)
ax.set_title('Top 15 words by count')
fig.savefig('./assets/top_15.png', bbox_inches='tight')



df_count = pd.DataFrame(X_count.toarray(), columns=count_vectorizer.get_feature_names_out())
df_count_sim = pd.DataFrame(cosine_similarity(df_count, dense_output=True))
df_count_sim_as_np = df_count_sim.values

np.fill_diagonal(df_count_sim_as_np, 0)
df_count_result = pd.DataFrame(df_count_sim_as_np)
df_count_result['best_match (count)'] = df_count_result.idxmax()
df_count_result['similarity (count)'] = df_count_result.max()

count_result = df_count_result[['best_match (count)', 'similarity (count)']]
df = df.merge(count_result, how='left', left_index=True, right_index=True)




# using tf-idf
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

df_tfidf_sim = pd.DataFrame(cosine_similarity(df_tfidf, dense_output=True))
df_tfidf_sim_as_np = df_tfidf_sim.values

np.fill_diagonal(df_tfidf_sim_as_np, 0)
df_tfidf_result = pd.DataFrame(df_tfidf_sim_as_np)
df_tfidf_result['best_match (tf-idf)'] = df_tfidf_result.idxmax()
df_tfidf_result['similarity (tf-idf)'] = df_tfidf_result.max()

tfidf_result = df_tfidf_result[['best_match (tf-idf)', 'similarity (tf-idf)']]
df = df.merge(tfidf_result, how='left', left_index=True, right_index=True)




df['best_match (count)'] = df['best_match (count)'].map(df['orig_thread'].to_dict())
df['best_match (tf-idf)'] = df['best_match (tf-idf)'].map(df['orig_thread'].to_dict())



# compare
# random 10
sample_index = np.random.randint(0, len(corpus), 10)

print(df.loc[sample_index][['orig_thread', 'best_match (count)', 'best_match (tf-idf)']].reset_index().to_markdown())



# plot
bins = [_/10 for _ in range(0, 11)]

fig, ax = plt.subplots()
ax.clear()
pd.cut(tfidf_result['similarity'], bins=bins).value_counts().sort_index().plot(ax=ax, kind='bar', width=1, ec='white', alpha=.3, color='#fb553b', label='tf-idf')
pd.cut(count_result['similarity'], bins=bins).value_counts().sort_index().plot(ax=ax, kind='bar', width=1, ec='white', alpha=.3, label='raw count')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
fig.savefig('./assets/compare sim score.png', bbox_inches='tight')