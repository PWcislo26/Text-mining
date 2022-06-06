import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import matplotlib.image

from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.corpus import stopwords


def cleanup_text(text: str) -> str:
    """cleaning input text"""
    emoticons = re.findall(r'[:|;][-]?[)|(|<>]', text)
    text_low = text.lower()
    text_without_number = re.sub(r'\d', '', text_low)
    text_without_html = re.sub(r'<.*?>', '', text_without_number)
    text_without_punc_marks = re.sub(r'\W(?<!\s)', '', text_without_html)
    text_without_white_space = text_without_punc_marks.strip()
    text_done = text_without_white_space + ' '.join(emoticons)
    return text_done


def delete_stop_words(text: str) -> list:
    """deleting stop words from input text"""
    stop_words = stopwords.words("english")
    return [w for w in text if not w.lower() in stop_words]


def stemming(word: str) -> str:
    ps = PorterStemmer()
    return ps.stem(word)


def text_tokenizer(text: str) -> list:
    cleaned = cleanup_text(text)
    tokens = word_tokenize(cleaned)
    without_stopwords = delete_stop_words(tokens)

    return [stemming(w) for w in without_stopwords if len(w) > 3]


def add_labels(x, y):
    """adding labels to bar chart """
    for i in range(1,len(x)+1):
        plt.text(i,y[i-1], y[i-1], ha="center", va="bottom")


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