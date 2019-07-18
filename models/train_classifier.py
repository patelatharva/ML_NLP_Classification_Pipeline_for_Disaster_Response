import sys
import pandas as pd
import numpy as np
from nltk import pos_tag
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet') # download for lemmatization
import string
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
# from sklearn.externals import joblib
import pickle
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM CategorizedMessages;", engine)
    print("Number of rows in df: ", df.shape[0])
    
    df[df.columns[4:]] = df[df.columns[4:]].astype(bool).astype(int)
    for column in df.columns[4:]:
        if df[column].sum() == 0:
            df = df.drop([column], axis=1)
    df.insert(loc=len(df.columns), column="unknown_category", value=0)
    df.loc[df[df.columns[4:]].sum(axis=1) == 0, "unknown_category"] = 1
    X = df["message"].values
    Y = df[df.columns[4:]]
    category_names = Y.columns
    return X, Y, category_names
    

def tokenize(text):
    def is_noun(tag):
        return tag in ['NN', 'NNS', 'NNP', 'NNPS']


    def is_verb(tag):
        return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


    def is_adverb(tag):
        return tag in ['RB', 'RBR', 'RBS']


    def is_adjective(tag):
        return tag in ['JJ', 'JJR', 'JJS']


    def penn_to_wn(tag):
        if is_adjective(tag):
            return wn.ADJ
        elif is_noun(tag):
            return wn.NOUN
        elif is_adverb(tag):
            return wn.ADV
        elif is_verb(tag):
            return wn.VERB
        return wn.NOUN
    
    words = word_tokenize(text.lower())
    tokens = [w for w in words if (w not in stopwords.words("english") and w not in string.punctuation)]
    tagged_words = pos_tag(tokens)
    lemmed = [WordNetLemmatizer().lemmatize(w.lower(), pos=penn_to_wn(tag)) for (w,tag) in tagged_words]
    if len(lemmed) == 0:
        return ["help"]
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        # ('clf', MultiOutputClassifier(estimator=SVC(gamma='scale')))
        ('clf', MultiOutputClassifier(estimator=MultinomialNB(fit_prior=False)))
    ])
    return pipeline
    # parameters = {
    #     'clf__estimator__fit_prior': [True, False]
    # }
    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultiOutputClassifier(estimator=SVC()))
    # ])
    # parameters = {'clf__estimator__gamma': [0.1, 1, 10, 50, 100],
    #             "clf__estimator__C": [0.1, 1, 10, 50, 100]}

    # cv = GridSearchCV(estimator=pipeline, param_grid=parameters, 
    #                   scoring='f1_weighted', n_jobs=1, verbose=1)
    # return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, model_filepath)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, open(model_filepath, 'wb'))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()