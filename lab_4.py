import re
from nltk import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


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
    clened = cleanup_text(text)
    tokens = word_tokenize(clened)
    without_stopwords = delete_stop_words(tokens)

    return [stemming(w) for w in without_stopwords if len(w) > 3]