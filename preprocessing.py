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
