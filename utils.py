import gc
import re
import os
import bz2
import pickle
from datetime import datetime, timedelta
from timeit import default_timer

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
    with Timer('Data Loading:'):
        df = pd.read_csv(os.path.join(directory, full_file_name), nrows=sample_size)
        df = reduce_mem_usage(df)
        gc.collect()
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


class Timer(object):
    """ A timer as a context manager
    Wraps around a timer. A custom timer can be passed
    to the constructor. The default timer is timeit.default_timer.
    Note that the latter measures wall clock time, not CPU time!
    On Unix systems, it corresponds to time.time.
    On Windows systems, it corresponds to time.clock.

    Adapted from: https://github.com/brouberol/contexttimer/blob/master/contexttimer/__init__.py

    Keyword arguments:
        output -- if True, print output after exiting context.
                  if callable, pass output to callable.
        format -- str.format string to be used for output; default "took {} seconds"
        prefix -- string to prepend (plus a space) to output
                  For convenience, if you only specify this, output defaults to True.
    """

    def __init__(self, prefix="", verbose=0, timer=default_timer):
        self.timer = timer
        self.verbose = verbose
        self.prefix = prefix
        self.end = None

    def __call__(self):
        """ Return the current time """
        return self.timer()

    def __enter__(self):
        """ Set the start time """
        self.start = self()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Set the end time """
        self.end = self()

        if self.verbose:
            message = " ".join([self.prefix, str(timedelta(seconds=self.elapsed))])
            if callable(self.verbose):
                self.verbose(message)
            else:
                print(message)
        gc.collect()

    def __str__(self):
        return str(timedelta(seconds=self.elapsed))

    @property
    def elapsed(self):
        """ Return the current elapsed time since start
        If the `elapsed` property is called in the context manager scope,
        the elapsed time bewteen start and property access is returned.
        However, if it is accessed outside of the context manager scope,
        it returns the elapsed time bewteen entering and exiting the scope.
        The `elapsed` property can thus be accessed at different points within
        the context manager scope, to time different parts of the block.
        """
        if self.end is None:
            # if elapsed is called in the context manager scope
            return self() - self.start
        else:
            # if elapsed is called out of the context manager scope
            return self.end - self.start


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


def save_model(step):
    current_datetime = datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
    str_valid_score = '{0:.5f}'.format(step['valid_score'])
    name = f'valid_score_{str_valid_score}__{current_datetime}'
    path = f'models/{name}.pkl.bz2'
    with bz2.BZ2File(path, 'w') as fout:
        pickle.dump(step, fout)
        step['cached'] = True
    step['path'] = path
    return step


def load_model(path):
    with bz2.BZ2File(path, 'r') as fin:
        return pickle.load(fin)


def predict_from_file(df, path):
    model = load_model(path)
    x = model['pipeline'].transform(df)
    return model['model'].predict(x)


def scores_postprocessing(df, predicted, columns, is_test=False):
    """
    TODO: разобраться че тут к чему и перепроверить код, скорее всего ошибся, когда переносил из GBM.ipynb
    :param df:
    :param predicted:
    :param columns:
    :param is_test:
    :return:
    """
    if is_test:
        df_sub = pd.read_csv('input/sample_submission_V2.csv', names=['id', 'win_place_perc'])
    else:
        df_sub = df.loc[:, ['id', ]].copy()
        df_sub[columns['target']] = predicted
    df_sub = df_sub.merge(df[["id", "match_id", "group_id", "max_place", "num_groups"]], on="id", how="left")

    # Sort, rank, and assign adjusted ratio
    df_sub_group = df_sub.groupby(["match_id", "group_id"]).first().reset_index()
    df_sub_group["rank"] = df_sub_group.groupby(["match_id"])["win_place_perc"].rank()
    df_sub_group = df_sub_group.merge(
        df_sub_group.groupby("match_id")["rank"].max().to_frame("max_rank").reset_index(),
        on="match_id", how="left")
    df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["num_groups"] - 1)

    df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "match_id", "group_id"]], on=["match_id", "group_id"],
                          how="left")
    df_sub["win_place_perc"] = df_sub["adjusted_perc"]

    # Deal with edge cases
    df_sub.loc[df_sub['max_place'] == 0, "win_place_perc"] = 0
    df_sub.loc[df_sub['max_place'] == 1, "win_place_perc"] = 1

    # Align with maxPlace
    # Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
    subset = df_sub.loc[df_sub['max_place'] > 1]
    gap = 1.0 / (subset['max_place'].values - 1)
    new_perc = np.around(subset['win_place_perc'].values / gap) * gap
    df_sub.loc[df_sub['max_place'] > 1, "win_place_perc"] = new_perc

    # Edge case
    df_sub.loc[(df_sub['max_place'] > 1) & (df_sub['num_groups'] == 1), "win_place_perc"] = 0
    assert df_sub["win_place_perc"].isnull().sum() == 0
    return df_sub[['id', 'win_place_perc']].copy()
