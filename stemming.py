import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

eee = """  Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's 
standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type
 specimen book. It has survived not only five centuries, but also the leap into electronic  """


def text_cleanup(text: str) -> str:
    remove_numbers = re.sub("[0-9]+", "", text)
    remove_html = re.sub("[^>]", "", remove_numbers)
    remove_punctuation = re.sub(r"[^\w\s]", '', remove_html)
    remove_whitespaces = re.sub(r"^\s+|\s+$",'', remove_punctuation)
    print(remove_numbers)


def stemming(text:str) -> list:
    pass


text_cleanup(eee)
remove_html = re.sub("[^>]", "", eee)
print(remove_html)