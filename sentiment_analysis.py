import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords


def generate_dataframe() -> pd.DataFrame:
    """Generate a dataframe from a csv file for further use"""
    df = pd.read_csv('amazon_alexa_reviews.csv', sep=';', usecols=['rating', 'verified_reviews'], encoding='cp1252')
    df['verified_reviews'].replace(' ', np.NaN, inplace=True)  # replace empty reviews with NaN
    df.dropna(how='any', inplace=True)  # delete reviews that are Nan
    df["sentiment"] = df.rating.apply(
        lambda x: 0 if x in [1, 2] else 1)  # add sentiment column, label reviews < 3 = 0, >=3 = 1
    return df


def show_plots(df: pd.DataFrame):
    """Generate plots related to dataframe"""
    ratings = df['rating'].value_counts()  # count distribution of review ratings
    plt.bar(ratings.index, ratings.values)
    plt.title("Distribution of review ratings")
    plt.xlabel("Review rating")
    plt.ylabel("Amount of reviews")
    plt.show()


def sentiment(df: pd.DataFrame):
    """Sentiment analysis for the dataframe"""
    x = df['verified_reviews']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    ctmTr = vectorizer.fit_transform(x_train)
    x_test_dtm = vectorizer.transform(x_test)
    lr = LogisticRegression()
    lr.fit(ctmTr, y_train)
    lr_score = lr.score(x_test_dtm, y_test)
    print(lr_score)
    y_pred_lr = lr.predict(x_test_dtm)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
    print(tn, fp, fn, tp)

    tpr_lr = round(tp / (tp + fn), 4)
    tnr_lr = round(tn / (tn + fp), 4)
    print(tpr_lr, tnr_lr)


# text_pos = " ".join(review for review in df.verified_reviews.astype(str))
# stop_list = set(stopwords.words('english'))
# wc = WordCloud(width=2500, height=2500, stopwords=stop_list, background_color='black', colormap='Paired')
# wc.generate(text_pos)
# wc.to_file('wc.png')

def main():
    df = generate_dataframe()
    show_plots(df)
    sentiment(df)


main()
