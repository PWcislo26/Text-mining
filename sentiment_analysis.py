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
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
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
    print(tabulate(sorted_tokens, headers='keys', tablefmt='psql'))
    plt.bar(sorted_tokens['Tokens'], sorted_tokens['Weights'])
    plt.xlabel("Tokens")
    plt.ylabel("Weights")
    plt.xticks(rotation=15)
    plt.gcf().subplots_adjust(bottom=0.15)
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
    print("Logistic Regression classification report")
    print(classification_report(y_test,y_pred_lr))
    """Precision informuje ile przypadków zdiagnozowanych pozytywnie jest rzeczywiscie pozytywna, dla klasy 0 precyzja 
     wynosi 0.89 a dla klasy 1 0.96, idealny wynik to 1. 
     Recall informuje o udziale przypadkow zdiagnozowanych pozytywnie wśród wszystkich przypadkow pozytywnych, 
     rowniez tych zaliczonych do negatywnych (FN). Dla klasy 0 recall wynosi 0.52
     a dla klasy 1 - 0.99. Niski wynik klasy 0 najprawdopodobniej wynika z bardzo małego udzialu ocen negatywnych w 
     zbiorze badanych danych, przez co algorytm nie zdołał odpowiednio nauczyć się wychwytawać tego typu przypadków.
     F1-score to srednia harmoniczna pomiędzy precision a recall, im blizej jest wartosci 1 tym lepszy algorytm 
     klasyfikujący dla danego przypadku, dla klasy 0, f1 score wynosi 0.66 a niski wynik najprawdopobniej związany jest
    z wyżej wymienionym powodem. Dla klasy 1 f1-score wynosi 0.98 co jest bardzo dobrym wynikiem.
    Dodatkowo z analizy wag tokenow w recenzjach negatywnych  i pozytywnych wynikało, że słowa nacechowane negatywnie 
    niekoniecznie są dominujące w recenzjach negatywnych, co dodatkowo utrudnia rozróżnienie ich od recenzji pozytwnych,
     z którym dzielone jest wiele słów neutralnych. 
    """
    confusion_table = texttable.Texttable()
    confusion_table.add_rows([["Confusion matrix results", "Number/ratio"],
                              ["True positives", tp],
                              ["True negatives", tn],
                              ["False positives", fp],
                              ["False negatives", fn],
                              ["True positives ratio", tp / (tp + fp)],
                              ["True negatives ratio", tn / (fn + tn)]])
    print(confusion_table.draw())

    svml = svm.SVC()
    svml = svml.fit(x_transform_train, y_train)
    svml_score = svml.score(x_transform_test,y_test)
    print(f"Support Vector Machine model prediction accuracy - {svml_score * 100} %")
    y_pred_svml = svml.predict(x_transform_test)
    print("Classification report for Support Vector Machine")
    print(classification_report(y_test, y_pred_svml)) #bardzo niski recall dla klasy 0, odrzucenie algorytmu
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_svml).ravel()
    confusion_table_svml = texttable.Texttable()
    confusion_table_svml.add_rows([["Confusion matrix results", "Number/ratio"],
                              ["True positives", tp],
                              ["True negatives", tn],
                              ["False positives", fp],
                              ["False negatives", fn],
                              ["True positives ratio", tp / (tp + fp)],
                              ["True negatives ratio", tn / (fn + tn)]])
    print(confusion_table_svml.draw())


    rfcl = RandomForestClassifier(n_estimators=150)
    rfcl =rfcl.fit(x_transform_train,y_train)
    rfcl_score = rfcl.score(x_transform_test,y_test)
    print(f"Random forest classifier prediction accuracy = {rfcl_score * 100} %")
    y_pred_rfcl= rfcl.predict(x_transform_test)
    print("Classification report for Random Forest Classifier")
    print(classification_report(y_test, y_pred_rfcl)) # recall < 0.5 dla klasy 0, odrzucenie algorytmu
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_rfcl).ravel()
    confusion_table_rfcl = texttable.Texttable()
    confusion_table_rfcl.add_rows([["Confusion matrix results", "Number/ratio"],
                                   ["True positives", tp],
                                   ["True negatives", tn],
                                   ["False positives", fp],
                                   ["False negatives", fn],
                                   ["True positives ratio", tp / (tp + fp)],
                                   ["True negatives ratio", tn / (fn + tn)]])
    print(confusion_table_rfcl.draw())


def main():
    df = generate_dataframe()
    show_plots(df)
    generate_wordclouds(df)
    show_wordclouds()
    token_weights(df)
    sentiment(df)


if __name__ == "__main__":
    main()
