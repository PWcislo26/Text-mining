import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

eee = """In Guam, she lAIDd in the sun 323321 and ate mangoes all day.
She got hiccups and couldn't get rid 22of them for three hours.
Let's all just takE a mOMent to breathe, please!"""


def text_cleanup(text: str) -> str:
    tmp = ""
    for char in text:
        if char.isalpha():
            tmp += char.lower()
        else:
            tmp += char
    remove_numbers = re.sub("[0-9]+", "", tmp)
    remove_punctuation = re.sub(r"[^\w\s]", '', remove_numbers)
    remove_whitespaces = re.sub(r"^\s+|\s+$", '', remove_punctuation)

    return remove_whitespaces


def stemming(text: str) -> list:
    ps = PorterStemmer()
    non_stopwords = []
    stemming_output = []
    stop_list = set(stopwords.words("english"))
    words = word_tokenize(text)
    for word in words:
        if word not in stop_list:
            non_stopwords.append(word)
    for word in non_stopwords:
        stemming_output.append(ps.stem(word))
    return stemming_output


print(stemming(text_cleanup(eee)))
