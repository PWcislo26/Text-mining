import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

example = """In Guam, she lAIDd in the sun 323321 and ate mangoes all day.
She got hiccups and couldn't get rid 22of them for three hours.
Let's all just takE a mOMent to breathe, please!"""


def text_cleanup(text: str) -> str:
    temp = re.sub(r"[:|;][-]?[\)|\(|<|>]", "", text)  # remove emotes
    temp = temp.lower()  # lower text
    temp = re.sub("\d", "", temp)  # remove digits
    temp = re.sub("http(s?)([^ ]*)", "", temp)  # remove links
    temp = re.sub(r"[^\w\s]", '', temp)  # remove punctuation marks
    temp = temp.strip()

    return temp


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


print(stemming(text_cleanup(example)))
