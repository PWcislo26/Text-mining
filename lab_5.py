import re

import numpy
import pandas
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


def cleanup_text(text: str) -> str:
    emoticons = re.findall(r'[:|;][-]?[)|(|<>]', text)
    text_low = text.lower()
    text_without_number = re.sub(r'\d', '', text_low)
    text_without_html = re.sub(r'<.*?>', '', text_without_number)
    text_without_punc_marks = re.sub(r'\W(?<!\s)', '', text_without_html)
    text_without_white_space = text_without_punc_marks.strip()
    text_done = text_without_white_space + ' '.join(emoticons)
    return text_done


def delete_stop_words(text: str) -> list:
    stop_words = stopwords.words("english")
    return [w for w in text if not w.lower() in stop_words]


def stemming(word: str) -> str:
    ps = PorterStemmer()
    return ps.stem(word)


def text_tokenizer(text: str):
    cleaned = cleanup_text(text)
    tokens = word_tokenize(cleaned)
    without_stopwords = delete_stop_words(tokens)

    return [stemming(w) for w in without_stopwords if len(w) > 3]


def tokens():
    """print 10 most occuring tokens from sample text"""
    reader = pandas.read_csv('True.csv', usecols=['title', 'text'], header=0)
    sample = reader['title'][:10]
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(sample)
    column_names = vectorizer.get_feature_names_out()  # token string representation
    array = x_transform.toarray()  # array of tokens
    column_sums = numpy.sum(array, axis=0)  # token occurrences
    most_occurring_idx = numpy.argpartition(column_sums, -10)[-10:]  # 10 most occurring token indexes
    most_occurring = []
    for index in numpy.nditer(most_occurring_idx):
        most_occurring.append(column_names[index])
    print(most_occurring)


def documents():
    reader = pandas.read_csv('True.csv', usecols=['title', 'text'], header=0)
    sample = reader['text'][:20]
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(sample)
    array = x_transform.toarray()
    print(array)
    row_sums = numpy.sum(array, axis=1)
    print(row_sums)
    most_tokens_idx = numpy.argpartition(row_sums, -10)[-10:]  # row indexes with most tokens
    print(most_tokens_idx)
    most_occurring = []
    for index in numpy.nditer(most_tokens_idx):  # prints documents with most tokens
        most_occurring.append(sample[index])
    print(most_occurring)


documents()
