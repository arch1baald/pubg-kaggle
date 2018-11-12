import sys
from types import ModuleType

class MockModule(ModuleType):
    def __init__(self, module_name, module_doc=None):
        ModuleType.__init__(self, module_name, module_doc)
        if '.' in module_name:
            package, module = module_name.rsplit('.', 1)
            get_mock_module(package).__path__ = []
            setattr(get_mock_module(package), module, self)

    def _initialize_(self, module_code):
        self.__dict__.update(module_code(self.__name__))
        self.__doc__ = module_code.__doc__

def get_mock_module(module_name):
    if module_name not in sys.modules:
        sys.modules[module_name] = MockModule(module_name)
    return sys.modules[module_name]

def modulize(module_name, dependencies=[]):
    for d in dependencies: get_mock_module(d)
    return get_mock_module(module_name)._initialize_

##===========================================================================##

@modulize('utils')
def _utils(__name__):
    ##----- Begin utils.py -------------------------------------------------------##
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
        abspath = os.path.abspath(path)
        with bz2.BZ2File(path, 'w') as fout:
            pickle.dump(step, fout)
            step['cached'] = True
        step['path'] = path
        step['abspath'] = abspath
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
    
    ##----- End utils.py ---------------------------------------------------------##
    return locals()

@modulize('features')
def _features(__name__):
    ##----- Begin features.py ----------------------------------------------------##
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    
    from utils import Timer
    
    
    class FeatureGenerator(BaseEstimator, TransformerMixin):
        """
        Hierarchy:
            - SimpleFeatureGenerator
            - GroupAggregatedFeatureGenerator, ... joined using FeatureUnion
            - ...
        """
        def __init__(self, numeric, id=None, target=None, categorical=None, verbose=0):
            self.created_features = None
            self.id = id
            self.target = target
            self.categorical = categorical
            self.numeric = numeric
            self.verbose = verbose
    
        def fit_transform(self, df, y=None, **fit_params):
            return self.transform(df)
    
        def transform(self, df):
            with Timer('features.FeatureGenerator.transform', verbose=self.verbose):
                # Hand Written Features
                simple_feature_generator = SimpleFeatureGenerator(numeric=self.numeric, verbose=self.verbose)
                df_features = pd.concat([df, simple_feature_generator.fit_transform(df)], axis=1)
    
                # 1-st level
                features = self.numeric + simple_feature_generator.get_feature_names()
                df_features = pd.concat([
                    df_features,
                    GroupAggregatedFeatureGenerator(features, verbose=self.verbose).fit_transform(df_features),
                ], axis=1)
    
                if self.created_features is None:
                    self.created_features = [col for col in df_features.columns if col in df.columns]
                else:
                    # TODO: test
                    pass
                return df_features
    
        def fit(self, x, y=None, **fit_params):
            return self
    
        def get_feature_names(self):
            return self.created_features
    
    
    class SimpleFeatureGenerator(BaseEstimator, TransformerMixin):
        """
        Used to create features via handwritten rules
    
        Based on https://www.kaggle.com/deffro/eda-is-fun
        """
        def __init__(self, numeric, id=None, target=None, categorical=None, verbose=0):
            self.created_features = None
            self.id = id
            self.target = target
            self.categorical = categorical
            self.numeric = numeric
            self.verbose = verbose
    
        def fit_transform(self, df, y=None, **fit_params):
            return self.transform(df)
    
        def transform(self, df):
            with Timer('features.SimpleFeatureGenerator.transform', verbose=self.verbose):
                df_features = pd.DataFrame()
                df_features['players_joined'] = df.groupby('match_id')['match_id'].transform('count')
                df_features['total_distance'] = df['ride_distance'] + df['walk_distance'] + df['swim_distance']
                df_features['kills_norm'] = df['kills'] * ((100 - df_features['players_joined']) / 100 + 1)
                df_features['damage_dealt_norm'] = df['damage_dealt'] * ((100 - df_features['players_joined']) / 100 + 1)
                df_features['heals_and_boosts'] = df['heals'] + df['boosts']
                df_features['total_distance'] = df['walk_distance'] + df['ride_distance'] + df['swim_distance']
                df_features['boosts_per_walk_distance'] = df['boosts'] / (df['walk_distance'] + 1)
                df_features['boosts_per_walk_distance'].fillna(0, inplace=True)
                df_features['heals_per_walk_distance'] = df['heals'] / (df['walk_distance'] + 1)
                df_features['heals_per_walk_distance'].fillna(0, inplace=True)
                df_features['heals_and_boosts_per_walk_distance'] = df_features['heals_and_boosts'] / (df['walk_distance'] + 1)
                df_features['heals_and_boosts_per_walk_distance'].fillna(0, inplace=True)
                df_features['kills_per_walk_distance'] = df['kills'] / (df['walk_distance'] + 1)
                df_features['kills_per_walk_distance'].fillna(0, inplace=True)
                df_features['team'] = [1 if i > 50 else 2 if (bool(i > 25) & bool(i <= 50)) else 4 for i in df['num_groups']]
    
                if self.created_features is None:
                    self.created_features = list(df_features.columns)
                else:
                    # TODO: test
                    pass
                return df_features
    
        def fit(self, x, y=None, **fit_params):
            return self
    
        def get_feature_names(self):
            return self.created_features
    
    
    class GroupAggregatedFeatureGenerator(BaseEstimator, TransformerMixin):
        """
        Used to create aggregations by categories
    
        Based on https://www.kaggle.com/anycode/simple-nn-baseline-4
        """
        def __init__(self, features, numeric=None, id=None, target=None, categorical=None, verbose=0):
            self.created_features = None
            self.id = id
            self.target = target
            self.categorical = categorical
            self.numeric = numeric
            self.verbose = verbose
    
            self.features = features
    
        def fit_transform(self, df, y=None, **fit_params):
            return self.transform(df)
    
        def transform(self, df):
            with Timer('features.GroupAggregatedFeatureGenerator.transform', verbose=self.verbose):
                df_features = []
                # Aggregate by Group
                # for agg_type in ('mean', 'max', 'min', 'count', 'std'):
                for agg_type in ('mean', 'max', 'min', 'count', ):
                    df_aggregated = df.groupby(['match_id', 'group_id'], as_index=False)[self.features].agg(agg_type)
                    df_aggregated = self.restore_row_order(df, df_aggregated, on=['match_id', 'group_id'])
                    agg_column_names = {col: f'{agg_type}_group_{col}' for col in self.features}
                    df_aggregated.rename(columns=agg_column_names, inplace=True)
    
                    # TODO: Computational problems
                    # # Rank Groups by Match
                    # columns_to_select = list(agg_column_names.values())
                    # # Anyway deletes match_id
                    # df_ranked = df_aggregated.groupby('match_id', as_index=False)[columns_to_select].rank(pct=True)
                    # ranked_column_names = {col: f'rank_{col}' for col in columns_to_select}
                    # df_ranked.rename(columns=ranked_column_names, inplace=True)
                    # # Unsafe merge because of rank, which deletes match_id
                    # df_aggregated_ranked = pd.concat([df_aggregated, df_ranked], axis=1)
                    # df_features.append(df_aggregated_ranked)
                    # del df_aggregated, df_ranked
                    df_features.append(df_aggregated)
                    del df_aggregated
                df_features = pd.concat(df_features, axis=1)
    
                if self.created_features is None:
                    self.created_features = list(df_features.columns)
                else:
                    if self.created_features == list(df_features.columns):
                        if self.verbose == 2:
                            print('Lost features')
                        for col in df_features.columns:
                            if col not in self.created_features:
                                if self.verbose == 2:
                                    print(col)
                return df_features
    
        def fit(self, x, y=None, **fit_params):
            return self
    
        def get_feature_names(self):
            return self.created_features
    
        def restore_row_order(self, df, df_aggregated, on):
            """
            Sometimes pd.merge shuffles rows.
            This merhod restores rows order after merging for correct FeatureUnion
            :param df: source DataFrame
            :param df_aggregated: DataFrame with new features
            :param on: columns list or str with column name
            :return: DataFrame with correct rows order
            """
            if isinstance(on, list):
                left_selected = ['index'] + on
            else:
                left_selected = ['index', on]
            df_features = df.reset_index()[left_selected].merge(
                df_aggregated,
                how='left',
                on=on,
            )
            df_features.set_index('index', inplace=True)
            df_features.sort_index(inplace=True)
            return df_features
    
    ##----- End features.py ------------------------------------------------------##
    return locals()

@modulize('preprocessing')
def _preprocessing(__name__):
    ##----- Begin preprocessing.py -----------------------------------------------##
    import numpy as np
    import pandas as pd
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import MinMaxScaler
    
    from utils import Timer
    
    
    class Preprocessor(BaseEstimator, TransformerMixin):
        """
        Preprocess data for training:
            - Drop columns
            - Fill missings
            - Normilize
            - ...
        """
        SELECTED_FEATURES = [
            'damage_dealt',
            'dbn_os',
            'kill_place',
            'kills',
            'longest_kill',
            'match_duration',
            'max_place',
            'num_groups',
            'walk_distance',
            'kills_norm',
            'damage_dealt_norm',
            'kills_per_walk_distance',
            'mean_group_boosts',
            'mean_group_damage_dealt',
            'mean_group_dbn_os',
            'mean_group_kill_place',
            'mean_group_kills',
            'mean_group_kill_streaks',
            'mean_group_longest_kill',
            'mean_group_match_duration',
            'mean_group_max_place',
            'mean_group_num_groups',
            'mean_group_walk_distance',
            'mean_group_total_distance',
            'mean_group_kills_norm',
            'mean_group_kills_per_walk_distance',
            'max_group_damage_dealt',
            'max_group_dbn_os',
            'max_group_kill_place',
            'max_group_kill_streaks',
            'max_group_longest_kill',
            'max_group_match_duration',
            'max_group_max_place',
            'max_group_num_groups',
            'max_group_walk_distance',
            'max_group_kills_norm',
            'max_group_damage_dealt_norm',
            'max_group_kills_per_walk_distance',
            'min_group_dbn_os',
            'min_group_kill_place',
            'min_group_kills',
            'min_group_kill_streaks',
            'min_group_longest_kill',
            'min_group_match_duration',
            'min_group_max_place',
            'min_group_num_groups',
            'min_group_walk_distance',
            'min_group_kills_norm',
            'min_group_damage_dealt_norm',
            'min_group_kills_per_walk_distance'
        ]
    
        def __init__(self, numeric, id=None, target=None, categorical=None, verbose=0):
            self.features = None
            self.id = id
            self.target = target
            self.categorical = categorical
            self.numeric = numeric
            self.verbose = verbose
    
            self.imputer = None
            self.scaler = None
    
        def fit_transform(self, df, y=None, **fit_params):
            """
            Used to train
            :param df: DataFrame
            :param y:
            :param fit_params:
            :return: DataFrame
            """
            with Timer('preprocessing.Preprocessor.fit_transform', verbose=self.verbose):
                # Drop columns
                to_drop = [
                    col
                    for col in df.columns
                    if col in self.id + [self.target] + self.categorical
                ]
                x = df.drop(to_drop, axis=1).copy()
    
                # Fill missings
                x.fillna(0, inplace=True)
    
                # Feature Selection
                non_selected = [col for col in x.columns if col not in self.SELECTED_FEATURES]
                x.drop(non_selected, axis=1, inplace=True)
    
                # Normilize
                self.scaler = MinMaxScaler()
                self.features = x.columns
                x = x.astype(np.float64)
                x = pd.DataFrame(self.scaler.fit_transform(x), columns=[
                    col for col in self.features if col in self.SELECTED_FEATURES])
                return x
    
        def transform(self, df):
            """
            Used to test/submit
            :param df: DataFrame
            :return: DataFrame
            """
            with Timer('preprocessing.Preprocessor.transform', verbose=self.verbose):
                # Drop ID and Categorical columns
                to_drop = [
                    col
                    for col in df.columns
                    if col in self.id + [self.target] + self.categorical
                ]
                x = df.drop(to_drop, axis=1).copy()
    
                # Feature Selection
                non_selected = [col for col in x.columns if col not in self.SELECTED_FEATURES]
                x.drop(non_selected, axis=1, inplace=True)
    
                # Fill missings
                x.fillna(0, inplace=True)
    
                # Normilize
                x = x.astype(np.float64)
                x = pd.DataFrame(self.scaler.transform(x), columns=[
                    col for col in self.features if col in self.SELECTED_FEATURES])
                return x
    
        def fit(self, x, y=None, **fit_params):
            return self
    
        def get_feature_names(self):
            return self.features
    
    ##----- End preprocessing.py -------------------------------------------------##
    return locals()

@modulize('pipelines')
def _pipelines(__name__):
    ##----- Begin pipelines.py ---------------------------------------------------##
    from sklearn.base import BaseEstimator, TransformerMixin
    
    from features import FeatureGenerator
    from preprocessing import Preprocessor
    from utils import Timer
    
    class NotFittedError(Exception):
        pass
    
    
    class Pipeline(BaseEstimator, TransformerMixin):
        """
        """
        def __init__(self, numeric, id=None, target=None, categorical=None, verbose=0):
            self.created_features = None
            self.id = id
            self.target = target
            self.categorical = categorical
            self.numeric = numeric
            self.verbose = verbose
    
            self.feature_generator = None
            self.preprocessor = None
    
        def fit_transform(self, df, y=None, **fit_params):
            with Timer('pipelines.Pipeline.fit_transform:', self.verbose):
                self.feature_generator = FeatureGenerator(
                    id=self.id,
                    numeric=self.numeric,
                    categorical=self.categorical,
                    target=self.target,
                    verbose=self.verbose,
                )
                df_features = self.feature_generator.fit_transform(df)
    
                self.preprocessor = Preprocessor(
                    id=self.id,
                    numeric=self.numeric,
                    categorical=self.categorical,
                    target=self.target,
                    verbose=self.verbose,
                )
                x = self.preprocessor.fit_transform(df_features)
                return x
    
        def transform(self, df):
            with Timer('pipelines.Pipeline.transform:', self.verbose):
                if self.feature_generator is None:
                    raise NotFittedError(f'feature_generator = {self.feature_generator}')
                if self.preprocessor is None:
                    raise NotFittedError(f'preprocessor = {self.preprocessor}')
    
                df_features = self.feature_generator.transform(df)
                x = self.preprocessor.transform(df_features)
                return x
    
        def fit(self, x, y=None, **fit_params):
            return self
    
        def get_feature_names(self):
            return self.created_features
    
    
    ##----- End pipelines.py -----------------------------------------------------##
    return locals()

@modulize('assess')
def _assess(__name__):
    ##----- Begin assess.py ------------------------------------------------------##
    import gc
    import os
    
    from pipelines import Pipeline
    from utils import (
        kfold_with_respect_to_groups, save_model, Timer, scores_postprocessing, load_data, split_columns_by_types
    )
    
    
    def assess(model, df, columns, metrics, n_splits=5, early_stopping_rounds=20, verbose=0):
        """
        k-fold cross-validation
    
        Checkpoints saving strategy ...
        :param model: sklearn-like object
        :param df: DataFrame with X and y
        :param columns: column names splited by types like utils.split_columns_by_types
        :param metrics: sklearn.metrics like function
        :param n_splits: the number of folds
        :param early_stopping_rounds: LightGBM param
        :param verbose: 0 - no logs, 1 - info, 2 - debug
        :return: iterations log
        """
        splits = kfold_with_respect_to_groups(df, n_splits=n_splits)
        log = []
        for train_index, valid_index in splits:
            print('\n---------------------------')
            with Timer('Data Preparation:', verbose):
                pipeline = Pipeline(**columns, verbose=verbose)
                x_train = pipeline.fit_transform(df.loc[train_index, :])
                y_train = df.loc[train_index, columns['target']]
                x_valid = pipeline.transform(df.loc[valid_index, :])
                y_valid = df.loc[valid_index, columns['target']]
    
            with Timer('Fitting:', verbose):
                model.fit(
                    x_train, y_train,
                    eval_set=[(x_valid, y_valid)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=-1 if verbose != 2 else 1,
                )
    
            # with Timer('Postprocessing:', verbose):
            #     pred_train = scores_postprocessing(
            #         df=df.loc[train_index, :],
            #         predicted=model.predict(x_train),
            #         columns=columns,
            #         is_test=False,
            #     )[columns['target']]
            #     pred_valid = scores_postprocessing(
            #         df=df.loc[valid_index, :],
            #         predicted=model.predict(x_valid),
            #         columns=columns,
            #         is_test=False,
            #     )[columns['target']]
            pred_train, pred_valid = model.predict(x_train), model.predict(x_valid)
    
            with Timer('Saving:', verbose):
                train_score = metrics(y_train, pred_train)
                valid_score = metrics(y_valid, pred_valid)
                step = dict(
                    model=model,
                    pipeline=pipeline,
                    train_score=train_score,
                    valid_score=valid_score,
                    not_adj_train_score=metrics(y_train, model.predict(x_train)),
                    not_adj_valid_score=metrics(y_valid, model.predict(x_valid)),
                    train_index=train_index,
                    valid_index=valid_index,
                    path=None,
                    cached=False,
                )
                try:
                    step = save_model(step)
                except Exception:
                    if verbose == 1:
                        print("Warning: Couldn't save the model")
                log.append(step)
                gc.collect()
    
            if verbose == 1:
                print(step['train_score'], step['valid_score'])
            print('---------------------------\n')
    
        if verbose == 1:
            print('Erasing cache ...')
        for idx, step in enumerate(sorted(log, key=lambda dct: dct['valid_score'], reverse=True)):
            if idx == 0:
                continue
            try:
                os.remove(step['path'])
                if verbose == 2:
                    print('Removed:', step['abspath'])
            except Exception:
                if verbose == 2:
                    print("Warning: Couldn't remove file:", step['abspath'])
        return log
    
    
    def main():
        from lightgbm import LGBMModel
        from sklearn.metrics import mean_absolute_error
    
        df = load_data('train', 'input', sample_size=10000)
        columns = split_columns_by_types(df)
        df.drop(df[df['win_place_perc'].isnull()].index, inplace=True)
        model_params = dict(
            objective='regression',
            metric='mae',
            n_jobs=-1,
            learning_rate=0.1,
            n_estimators=2000,
        )
        assessment_log = assess(
            LGBMModel(**model_params),
            df,
            columns,
            metrics=mean_absolute_error,
            n_splits=5,
            early_stopping_rounds=20,
            verbose=1,
        )
    
    ##----- End assess.py --------------------------------------------------------##
    return locals()

from assess import main
main()
