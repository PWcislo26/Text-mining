import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import texttable

from util import text_tokenizer, generate_dataframe, generate_wordclouds, show_wordclouds, show_plots
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate


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
    """Sentiment analysis for the amazon alexa review data with Logistic Regression,
    RandomForestClassifier and Support Vector Machine"""
    x = df['verified_reviews']
    y = df['sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
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
    print(classification_report(y_test, y_pred_lr))
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
    svml_score = svml.score(x_transform_test, y_test)
    print(f"Support Vector Machine model prediction accuracy - {svml_score * 100} %")
    y_pred_svml = svml.predict(x_transform_test)
    print("Classification report for Support Vector Machine")
    print(classification_report(y_test, y_pred_svml))  # bardzo niski recall dla klasy 0, odrzucenie algorytmu
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
    rfcl = rfcl.fit(x_transform_train, y_train)
    rfcl_score = rfcl.score(x_transform_test, y_test)
    print(f"Random forest classifier prediction accuracy = {rfcl_score * 100} %")
    y_pred_rfcl = rfcl.predict(x_transform_test)
    print("Classification report for Random Forest Classifier")
    print(classification_report(y_test, y_pred_rfcl))  # recall < 0.5 dla klasy 0, odrzucenie algorytmu
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
