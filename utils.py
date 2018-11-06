import re
import os
import bz2
import pickle
from datetime import datetime

import pandas as pd


def camelcase_to_underscore(string):
    """
    Used to rename columns like MatchId -> match_id
    :param string: CamelCase string
    :return: _underscore string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_data(file_name, directory='../input/', nrows=None):
    """
    Load data from .csv file.
    Transform columns names from CamelCase to _underscore notation.
    :param file_name: file name
    :param directory: path to the directory with the file
    :param nrows: sample size
    :return: DataFrame
    """
    if file_name.startswith('train'):
        full_file_name = 'train_V2.csv'
    elif file_name.startswith('test'):
        full_file_name = 'test_V2.csv'
    elif 'submission' in file_name:
        full_file_name = 'sample_submission_V2.csv'
    else:
        full_file_name = file_name
    df = pd.read_csv(os.path.join(directory, full_file_name), nrows=nrows)
    df.columns = [camelcase_to_underscore(col) for col in df.columns]
    return df


def split_columns_by_types(df):
    """
    Names of ID, Categorical, Numeric and Target columns.
    :param df: DataFrame
    :return: dict
    """
    id_columns = ['id', 'group_id', 'match_id']
    categorical_columns = ['match_type', ]
    target_column = 'win_place_perc'
    numeric_columns = [
        col
        for col in df.columns
        if col not in id_columns + categorical_columns + [target_column]
    ]
    return dict(
        id=id_columns,
        target=target_column,
        categorical=categorical_columns,
        numeric=numeric_columns
    )


def save_model(pipeline_model_index_score):
    current_datetime = datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
    str_valid_score = '{0:.5f}'.format(pipeline_model_index_score['valid_score'])
    name = f'valid_score_{str_valid_score}__{current_datetime}'
    path = f'models/{name}.pkl.bz2'
    with bz2.BZ2File(path, 'w') as fout:
        pickle.dump(pipeline_model_index_score, fout)


def load_model(path):
    with bz2.BZ2File(path, 'r') as fin:
        return pickle.load(fin)


def predict_from_file(df, path):
    model = load_model(path)
    x = model['pipeline'].transform(df)
    return model['model'].predict(x)
