import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import text_tokenizer, generate_dataframe, generate_wordclouds, show_wordclouds, show_plots
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tabulate import tabulate


def token_weights_positive(df: pd.DataFrame):
    """Show 10 highest token weights for positive reviews"""
    df_pos = df[df['airline_sentiment'] == "positive"]
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform_pos = vectorizer.fit_transform(df_pos['text'])
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
    print("\n 10 most important tokens for positive reviews")
    print(tabulate(sorted_tokens, headers='keys', tablefmt='psql'))
    plt.bar(sorted_tokens['Tokens'], sorted_tokens['Weights'])
    plt.xlabel("Tokens")
    plt.ylabel("Weights")
    plt.xticks(rotation=25)
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.title("10 most important tokens for positive reviews")
    plt.show()


def token_weights_negative(df: pd.DataFrame):
    """Show 10 highest token weights for negative reviews"""
    df_neg = df[df['airline_sentiment'] == "negative"]
    vectorizer = TfidfVectorizer(tokenizer=text_tokenizer)
    x_transform_neg = vectorizer.fit_transform(df_neg['text'])
    column_names_neg = vectorizer.get_feature_names_out()
    array_neg = x_transform_neg.toarray()
    token_column_sums_neg = np.sum(array_neg, axis=0)
    highest_weight_indexes_neg = np.argpartition(token_column_sums_neg, -10)[-10:]
    highest_weight_token_names_neg = []
    highest_weight_neg = []

    for index in np.nditer(highest_weight_indexes_neg):
        highest_weight_token_names_neg.append((column_names_neg[index]))
        highest_weight_neg.append(token_column_sums_neg[index])

    data_neg = {'Tokens': highest_weight_token_names_neg, 'Weights': highest_weight_neg}
    tokens_neg = pd.DataFrame(data_neg)
    sorted_tokens = tokens_neg.sort_values(by=['Weights'], ascending=True)
    print("\n 10 most important tokens for negative reviews")
    print(tabulate(sorted_tokens, headers='keys', tablefmt='psql'))
    plt.bar(sorted_tokens['Tokens'], sorted_tokens['Weights'])
    plt.xlabel("Tokens")
    plt.ylabel("Weights")
    plt.xticks(rotation=25)
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.title("10 most important tokens for negative reviews")
    plt.show()


def sentiment(df: pd.DataFrame):
    """Sentiment analysis for the amazon alexa review data with Logistic Regression,
    RandomForestClassifier and Support Vector Machine"""
    x = df['text']
    y = df['airline_sentiment']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    vectorizer = CountVectorizer(tokenizer=text_tokenizer)
    x_transform_train = vectorizer.fit_transform(x_train)
    x_transform_test = vectorizer.transform(x_test)
    dt = DecisionTreeClassifier()
    dt.fit(x_transform_train, y_train)
    dt_score = dt.score(x_transform_test, y_test)
    print(f"Decision Tree model prediction accuracy - {dt_score * 100} %")
    y_pred_dt = dt.predict(x_transform_test)
    print("Decision Tree classification report")
    print(classification_report(y_test, y_pred_dt))

    svml = svm.SVC()
    svml = svml.fit(x_transform_train, y_train)
    svml_score = svml.score(x_transform_test, y_test)
    print(f"Support Vector Machine model prediction accuracy - {svml_score * 100} %")
    y_pred_svml = svml.predict(x_transform_test)
    print("Classification report for Support Vector Machine")
    print(classification_report(y_test, y_pred_svml))
    """Klasyfikator przewiduje sentyment tweetow z trafnością ok. 78.5%.
    Precision informuje jaki odsetek przypadków zdiagnozowanych pozytywnie/neturalnie/negatywnie rzeczywiscie taki jest.
    Dla recenzji negatywnych precision wynosi 0.82, warto zwrocic takze uwage ze liczba ocen negatywnych jest znaczaco
    przeważająca. Precision dla ocen neutralnych wynosi 0.63, moze to wynikac z uzycia słow neutralnych, które często
    będą wykorzystywane także w recenzjach pozytywnych i negatywnych. Precision dla recenzji pozytywnych wynosi 0.78.
    Poza recenzjami neutralnymi klasyfikator radzi sobie w miare dobrze w klasyfikacji recenzji.
    Recall informuje o udziale przypadkow zdiagnozowanych pozytwnie/neutralnie/negatywnie wśród wszystkich 
    takich przypadków, rowniez tych zaliczonych do innych klas (FN). Recall dla klas negatywyny, neutralny, pozytywny
    wynosi odpowiednio 0.92, 0.49, 0.62. Znów można wyróżnić słaby wynik recenzji neutralnych, jest to 
    najprawdopodobniej spowodowane wcześniej wymienionymi przyczynami. F1-score to srednia harmoniczna pomiędzy 
    precision a recall, im blizej jest wartosci 1 tym lepszy algorytm klasyfikujący dla danego przypadku.
     F1 dla negatywnych wynosi 0.87 jest do dobry wynik, dla neutralnych 0.55 a dla pozytywnych 0.69.
     Warto zauważyć również, że pomimo najmniejszej liczby recenzji pozytywnych klasyfikator radzi sobie znacznie lepiej
     z ich wykyrwaniem niż w przypadku recenzji neutralnych."""

    rfcl = RandomForestClassifier(n_estimators=150)
    rfcl = rfcl.fit(x_transform_train, y_train)
    rfcl_score = rfcl.score(x_transform_test, y_test)
    print(f"Random forest classifier prediction accuracy = {rfcl_score * 100} %")
    y_pred_rfcl = rfcl.predict(x_transform_test)
    print("Classification report for Random Forest Classifier")
    print(classification_report(y_test, y_pred_rfcl))


def main():
    df = generate_dataframe()
    show_plots(df)
    generate_wordclouds(df)
    show_wordclouds()
    token_weights_positive(df)
    token_weights_negative(df)
    sentiment(df)


if __name__ == "__main__":
    main()
