from bs4 import BeautifulSoup
from konlpy.tag import Kkma, Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from pandas  import DataFrame, Series
import pandas as pd
from newspaper import Article

url_list = ['http://v.media.daum.net/v/20171215130602344',
            'http://v.media.daum.net/v/20171215130312300',
            'http://v.media.daum.net/v/20171215111203921',
            'http://v.media.daum.net/v/20171216002700566',
            'http://v.media.daum.net/v/20171215214505350']

def TextSimilarity (url_list):

    okt = Okt()
    # kkma = Kkma()

    mydoclist_okt = []
    # mydoclist_kkma = []

    for url in url_list:
      article = Article(url, langague='ko')
      article.download()
      article.parse()
    
      # kkma_nouns = ' '.join(kkma.nouns(text))
      okt_nouns = ' '.join(okt.nouns(article.text))
      # mydoclist_kkma.append(kkma_nouns)
      mydoclist_okt.append(okt_nouns)

    tfidf_vectorizer = TfidfVectorizer(min_df = 1)
    # tfidf_matrix_kkma = tfidf_vectorizer.fit_transform(mydoclist_kkma)
    tfidf_matrix_okt = tfidf_vectorizer.fit_transform(mydoclist_okt)

    # document_distances_kkma = (tfidf_matrix_kkma * tfidf_matrix_kkma.T)
    document_distances_okt = (tfidf_matrix_okt * tfidf_matrix_okt.T)

    print(document_distances_okt.toarray())
    df = DataFrame(document_distances_okt.toarray(), columns=url_list, index=url_list)
    plt.figure(figsize=(document_distances_okt.get_shape()[0],document_distances_okt.get_shape()[0]))
    sns.heatmap(data = df, annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')
    plt.title('similarity analysis', fontsize=20)
    plt.show()

TextSimilarity(url_list)