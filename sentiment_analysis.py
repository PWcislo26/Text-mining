import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import text_tokenizer, addlabels
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
    ratings = df['rating'].value_counts().sort_index()  # count distribution of review ratings
    plt.bar(ratings.index, ratings.values)
    plt.title(f"Distribution of review ratings")
    plt.xlabel("Review rating")
    plt.ylabel("Number of reviews")
    addlabels(ratings.index, ratings.values)
    plt.show()
    print(ratings)

    sentiments = df['sentiment'].value_counts()
    plt.pie(sentiments.values, shadow=True, labels=["Positive", "Negative"], startangle=90, autopct='%1.1f%%',
            colors=["Green", "Red"])
    plt.title("Distribution of negative and positive sentiment reviews")
    for i in range(len(ratings.index)):
        plt.text(i, ratings.values[i],ratings.values[i], ha="center", va="bottom")
    plt.show()


def sentiment(df: pd.DataFrame):
    """Sentiment analysis for the dataframe"""
    X = df['verified_reviews']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform_train = vectorizer.fit_transform(X_train)
    x_transform_test = vectorizer.transform(X_test)
    lr = LogisticRegression()
    lr.fit(x_transform_train, y_train)
    lr_score = lr.score(x_transform_test, y_test)
    print(f"Logistic regression prediction accuracy - {round(lr_score,3) * 100} %")
    y_pred_lr = lr.predict(x_transform_train)


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
