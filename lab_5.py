import numpy
import pandas
import matplotlib.pyplot as plt

from tools import text_tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tabulate import tabulate


def tokens():
    """returns a dataframe of 10 most occurring tokens """
    reader = pandas.read_csv('True.csv', usecols=['title', 'text'], header=0)
    sample = reader['text'][:10]
    vectorizer_tokens = CountVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer_tokens.fit_transform(sample)
    column_names = vectorizer_tokens.get_feature_names_out()  # token string representation
    array = x_transform.toarray()  # array of tokens
    column_sums = numpy.sum(array, axis=0)  # token occurrences
    most_occurring_idx = numpy.argpartition(column_sums, -10)[-10:]  # 10 most occurring token indexes
    most_occurring_tokens = []
    token_occurrences = []

    for index in numpy.nditer(most_occurring_idx):
        most_occurring_tokens.append(column_names[index])
        token_occurrences.append(column_sums[index])

    data = {'Tokens': most_occurring_tokens, 'Appearances': token_occurrences}
    df = pandas.DataFrame(data)
    return df


def token_weights():
    """returns a dataframe of 10 tokens with highest combined weights"""
    reader = pandas.read_csv('True.csv', usecols=['title', 'text'], header=0)
    sample = reader['text'][:10]
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform = vectorizer.fit_transform(sample)
    column_names = vectorizer.get_feature_names_out()  # token string representation
    array = x_transform.toarray()  # array of tokens
    token_column_sums = numpy.sum(array, axis=0)  # summed token weights
    highest_weight_indexes = numpy.argpartition(token_column_sums, -10)[-10:]  # 10 token indexes with highest summed weight
    highest_weight_tokens = []  # 10 highest token weight names
    highest_weight = []  # 10 highest weights

    for index in numpy.nditer(highest_weight_indexes):
        highest_weight_tokens.append((column_names[index]))
        highest_weight.append(token_column_sums[index])

    data = {'Tokens': highest_weight_tokens, 'Weights': highest_weight}
    df = pandas.DataFrame(data)
    return df


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

    for index in numpy.nditer(most_tokens_idx):
        most_occurring.append(sample[index])
    print(most_occurring)


def main():
    data_tokens = tokens()
    data_tokens.plot(kind='barh', x='Tokens', y='Appearances')
    print(tabulate(data_tokens, headers='keys', tablefmt='psql'))
    plt.show()

    data_weights = token_weights()
    data_weights.plot(kind='barh', x='Tokens', y='Weights')
    print(tabulate(data_weights, headers='keys', tablefmt='psql'))
    plt.show()


main()
