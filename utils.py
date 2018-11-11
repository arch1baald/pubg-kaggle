import re
import os
import bz2
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def camelcase_to_underscore(string):
    """
    Used to rename columns like MatchId -> match_id
    :param string: CamelCase string
    :return: _underscore string
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def load_data(file_name, directory='../input/', sample_size=None):
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
    print('Loading ...')
    df = pd.read_csv(os.path.join(directory, full_file_name), nrows=sample_size)
    print('Compressing ...')
    df = reduce_mem_usage(df)
    df.columns = [camelcase_to_underscore(col) for col in df.columns]
    return df


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage

    Source: https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
    :param df: DataFrame
    :return:
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))
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


def kfold_with_respect_to_groups(df, n_splits, shuffle=True, random_state=None):
    """
    Splits data with respect to groups in matches.
    To apply adjustment trick, players of one group in matches have to fall in the same fold.
    :param df: DataFrame
    :param n_splits: the number of folds
    :param shuffle:
    :param random_state:
    :return: splits = [(train_idx, test_idx), ..., (train_idx, test_idx)]
    """
    df_match_groups = df.groupby(['match_id', 'group_id'], as_index=False)['id'].count()

    kfold = KFold(n_splits, shuffle, random_state)
    splits = []
    for pseudo_train_idx, pseudo_valid_idx in kfold.split(df_match_groups):
        df_train_match_groups = df_match_groups.loc[pseudo_train_idx, :]
        select_train_match_groups = (
                df['match_id'].isin(df_train_match_groups['match_id'])
                & df['group_id'].isin(df_train_match_groups['group_id'])
        )
        train_idx = df[select_train_match_groups].index

        df_valid_match_groups = df_match_groups.loc[pseudo_valid_idx, :]
        select_valid_match_groups = (
                df['match_id'].isin(df_valid_match_groups['match_id'])
                & df['group_id'].isin(df_valid_match_groups['group_id'])
        )
        valid_idx = df[select_valid_match_groups].index
        splits.append((train_idx, valid_idx))
    return splits


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
