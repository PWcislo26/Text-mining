import re
import matplotlib.pyplot as plt
import pandas as pd
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
    for i in range(1, len(x) + 1):
        plt.text(i, y[i - 1], y[i - 1], ha="center", va="bottom")


def generate_dataframe() -> pd.DataFrame:
    """Generate a dataframe from a csv file for further use"""
    df = pd.read_csv('tweets_airline.csv', sep=',',
                     usecols=['text', 'airline_sentiment'], encoding='utf-8')
    return df


def show_plots(df: pd.DataFrame):
    """Generate plots related to dataframe"""
    ratings = df['airline_sentiment'].value_counts().sort_index()  # count distribution of review ratings
    plt.bar(ratings.index, ratings.values)
    plt.title(f"Distribution of review ratings")
    plt.xlabel("Review rating score")
    plt.ylabel("Number of reviews")
    plt.show()

    sentiments = df['airline_sentiment'].value_counts()
    plt.pie(sentiments.values, shadow=True, labels=["Negative", "Neutral", "Positive"],
            startangle=90, autopct='%1.1f%%',
            colors=["Red", "Blue", "Green"])
    plt.title("Distribution of negative, neutral and positve airline reviews")
    plt.show()


def generate_wordclouds(df: pd.DataFrame):
    """Generate wordclouds for all reviews, positive only and negative only"""
    stop_list = set(stopwords.words('english'))
    text_general = " ".join(review for review in df.text.astype(str))
    text_general = cleanup_text(text_general)
    wc = WordCloud(width=2500, height=2500, stopwords=stop_list, background_color='black', colormap='Paired')
    wc.generate(text_general)
    wc.to_file('wordclouds/wc_general.png')


def show_wordclouds():
    """Show generated wordclouds from wordclouds directory"""
    for path in glob.glob('wordclouds/*'):
        plt.imshow(matplotlib.image.imread(path))
        plt.title(path.split('\\')[1])
        plt.axis("off")
        plt.show()
