#!/usr/bin/env python

import re
from itertools import product, chain
import warnings

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn_pandas import DataFrameMapper, gen_features
import sklearn
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, confusion_matrix, make_scorer
from sklearn.ensemble import VotingClassifier

import xgboost

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

np.random.seed(0)
stop = stopwords.words('english')

def init_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')

def build_pipeline():
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(course_df['Title'])


def clean_str_col(df, col_names):
    """Removes stop words and other extra characters
        df: DataFrame
        col_names: list of strs
    """

    df = df[col_names].copy()
    df = df.applymap(lambda x: re.sub('[^a-zA-Z ]+', '', x.replace('\"', '').lower()))
    return df.applymap(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))


def clean_csc_courses():
    course_df = pd.DataFrame(pd.read_csv('ncsu_course_info.csv', header=None).values.reshape((-1, 2)), columns = ['Title', 'Description'])
    course_df[['Title', 'Description']] = clean_str_col(course_df, ['Title', 'Description'])
    course_df.columns = ['Name', 'Desc']
    course_df.to_csv('clean_courses.csv', index=0)


def clean_software_desc():
    soft_df = pd.read_csv('soft_descs.csv')
    soft_df[['Desc']] = clean_str_col(soft_df, ['Desc'])
    soft_df.to_csv('clean_soft_descs.csv', index=0)

def generate_raw_train():
    soft_df = pd.read_csv('clean_soft_descs.csv')
    course_df = pd.read_csv('clean_courses_labeled.csv')

    soft_cols = [s+'_soft' for s in soft_df.columns]
    course_cols = [c+'_course' for c in course_df.columns]

    soft_list = soft_df.values.tolist()
    course_list = course_df.values.tolist()

    out_df = pd.DataFrame(
        [list(chain(*r)) for r in list(product(soft_list, course_list))],
        columns=soft_cols+course_cols
    )
    out_df['Label'] = (out_df['Name_soft'] == out_df['Label_course']).astype(int)
    out_df.drop(columns=['Name_soft', 'Label_course'], inplace=True)
    out_df.columns = ['soft_desc', 'course_name', 'course_desc', 'label']
    out_df.to_csv('raw_ml_construction.csv', index=0)

def build_model():
    data_df = pd.read_csv('raw_ml_construction.csv')

    x_data_df = data_df.drop(columns='label')
    y_data_df = data_df[['label']]

    print(y_data_df['label'].value_counts())

    # X_train, X_test, y_train, y_test = train_test_split(x_data_df,
    #     y_data_df, test_size=0.2)

    tf_idf_args = {
        'sublinear_tf': True,
        'max_df': 0.5,
        'norm': 'l2',
    }

    data_mapper = DataFrameMapper(
        [
            ("soft_desc", sklearn.feature_extraction.text.TfidfVectorizer(**tf_idf_args, ngram_range=(2, 4), max_features=300)),
            ("course_name", sklearn.feature_extraction.text.TfidfVectorizer(**tf_idf_args, ngram_range=(2, 2), max_features=50)),
            ("course_desc", sklearn.feature_extraction.text.TfidfVectorizer(**tf_idf_args, ngram_range=(2, 4), max_features=300)),
        ],
        df_out=True,
    )

    # X_train = data_mapper.fit_transform(X_train)
    # selected_cols = X_train.columns
    # X_train = X_train.values
    # X_test = data_mapper.transform(X_test).values

    X_data = data_mapper.fit_transform(x_data_df)
    y_data = y_data_df

    model = xgboost.XGBClassifier(
        max_depth = 12,
        subsample = 0.8, 
        scale_pos_weight = 9
    )

    f1 = make_scorer(f1_score)
    cv_scores = cross_validate(
        model,
        X_data,
        y_data,
        scoring=f1,
        cv=3,
        return_train_score=False,
        return_estimator=True,
        # fit_params=fit_params
    )

    test_cv_scores = cv_scores["test_score"]
    test_cv_summary = (cv_scores["test_score"].mean(), cv_scores["test_score"].std())
    print('Cross val scores: ', ['{:.3f}'.format(x) for x in test_cv_scores])
    print('Mean and Std: ', ['{:.3f}'.format(x) for x in test_cv_summary])

    trained_estimator = cv_scores['estimator']

    final_model = VotingClassifier(estimators=[(str(i), m) for i, m in enumerate(trained_estimator)], voting='hard')

    y_pred = cross_val_predict(final_model, X_data, y_data, cv=3)
    print('Confusion Matrix')
    cm = confusion_matrix(y_data, y_pred)
    print(cm)

    f, ax = plt.subplots(figsize=[7,10])
    
    xgboost.plot_importance(trained_estimator[0], max_num_features=50, ax=ax)
    plt.title("XGBOOST Feature Importance")

    plt.figure(figsize = (10,7))
    sn.heatmap(cm)

    plt.show()




if __name__ == '__main__':
    clean_csc_courses()
    clean_software_desc()
    generate_raw_train()
    build_model()
