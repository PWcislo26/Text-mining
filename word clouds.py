import csv
import re

from wordcloud import WordCloud


def text_cleanup(text):
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


with open('True.csv', newline='', encoding="utf8") as text_true:
    reader = csv.reader(text_true)
    for row in reader:
        print(row[0:-2])
