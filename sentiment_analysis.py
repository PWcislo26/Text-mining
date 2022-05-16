import matplotlib.image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import texttable
import glob

from util import text_tokenizer, add_labels, cleanup_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
from tabulate import tabulate

def generate_dataframe() -> pd.DataFrame:
    """Generate a dataframe from a csv file for further use"""
    df = pd.read_csv('data/amazon_alexa_reviews.csv', sep=';',
                     usecols=['rating', 'verified_reviews'], encoding='cp1252')
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
    plt.xlabel("Review rating score")
    plt.ylabel("Number of reviews")
    add_labels(ratings.index, ratings.values)
    plt.show()

    sentiments = df['sentiment'].value_counts()  # count distribution of positive and negative sentiment among reviews
    plt.pie(sentiments.values, shadow=True, labels=["Positive", "Negative"], startangle=90, autopct='%1.1f%%',
            colors=["Green", "Red"])
    plt.title("Distribution of positive and negative sentiment reviews")
    plt.show()


def generate_wordclouds(df: pd.DataFrame):
    """Generate wordclouds for all reviews, positive only and negative only"""
    stop_list = set(stopwords.words('english'))
    text_general = " ".join(review for review in df.verified_reviews.astype(str))
    text_general = cleanup_text(text_general)
    wc = WordCloud(width=2500, height=2500, stopwords=stop_list, background_color='black', colormap='Paired')
    wc.generate(text_general)
    wc.to_file('wordclouds/wc_general.png')

    df_pos = df[df['rating'] >= 3]
    text_pos = " ".join(review for review in df_pos.verified_reviews.astype(str))
    text_pos = cleanup_text(text_pos)
    wc.generate(text_pos)
    wc.to_file('wordclouds/wc_positive.png')

    df_neg = df[df['rating'] < 3]
    text_neg = " ".join(review for review in df_neg.verified_reviews.astype(str))
    text_neg = cleanup_text(text_neg)
    wc.generate(text_neg)
    wc.to_file('wordclouds/wc_negative.png')


def show_wordclouds():
    """Show generated wordclouds from wordclouds directory"""
    for path in glob.glob('wordclouds/*'):
        plt.imshow(matplotlib.image.imread(path))
        plt.title(path.split('\\')[1])
        plt.axis("off")
        plt.show()


def token_weights(df: pd.DataFrame):
    """Show 10 highest token weights for positive reviews"""
    df_pos = df[df['rating'] > 2]
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform_pos = vectorizer.fit_transform(df_pos['verified_reviews'])
    column_names_pos = vectorizer.get_feature_names_out()
    array_pos = x_transform_pos.toarray()
    token_column_sums_pos = np.sum(array_pos, axis=0)
    highest_weight_indexes_pos = np.argpartition(token_column_sums_pos, -10)[-10:]
    highest_weight_token_names_pos = []
    highest_weight_pos = []

    for index in np.nditer(highest_weight_indexes_pos):
        highest_weight_token_names_pos.append((column_names_pos[index]))
        highest_weight_pos.append(token_column_sums_pos[index])

    data_pos = {'Tokens': highest_weight_token_names_pos, 'Weights': highest_weight_pos}
    tokens_pos = pd.DataFrame(data_pos)
    sorted_tokens = tokens_pos.sort_values(by=['Weights'], ascending=True)
    sorted_tokens.plot(kind='bar', x='Tokens', y='Weights')
    plt.title("10 most important tokens for positive reviews")
    plt.show()


def sentiment(df: pd.DataFrame):
    """Sentiment analysis for the amazon alexa review data with logistic regression prediction machine learning"""
    x = df['verified_reviews']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform_train = vectorizer.fit_transform(x_train)
    x_transform_test = vectorizer.transform(x_test)
    lr = LogisticRegression()
    lr.fit(x_transform_train, y_train)
    lr_score = lr.score(x_transform_test, y_test)
    print(f"Logistic regression model prediction accuracy - {lr_score * 100} %")
    y_pred_lr = lr.predict(x_transform_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
    confusion_table = texttable.Texttable()
    confusion_table.add_rows([["Confusion matrix results", "Number/ratio"],
                              ["True positives", tp],
                              ["True negatives", tn],
                              ["False positives", fp],
                              ["False negatives", fn],
                              ["True positives ratio", tp / (tp + fp)],
                              ["True negatives ratio", tn / (fn + tn)]])
    print(confusion_table.draw())


def main():
    df = generate_dataframe()
    show_plots(df)
    generate_wordclouds(df)
    show_wordclouds()
    token_weights(df)
    sentiment(df)


if __name__ == "__main__":
    main()
