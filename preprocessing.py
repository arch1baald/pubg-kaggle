import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


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


class Preprocessor(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, numerical_columns, id_columns=None, target_column=None, categorical_columns=None):
        self.features = None
        self.id_columns = id_columns
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.imputer = None
        self.scaler = None

    def fit_transform(self, df, y=None, **fit_params):
        print('Preprocessor ...')
        # Drop columns
        x = df.drop(self.id_columns + [self.target_column] + self.categorical_columns, axis=1).copy()
        # Fill missings
        x.fillna(0, inplace=True)
        # Feature Selection
        non_selected = [col for col in x.columns if col not in SELECTED_FEATURES]
        x.drop(non_selected, axis=1, inplace=True)
        # Normilize
        self.scaler = MinMaxScaler()
        self.features = x.columns
        x = x.astype(np.float64)
        x = pd.DataFrame(self.scaler.fit_transform(x), columns=[col for col in self.features if col in SELECTED_FEATURES])
        return x


    def transform(self, df):
        print('Preprocessor ...')
        # Drop columns
        x = df.drop(self.id_columns + [self.target_column] + self.categorical_columns, axis=1).copy()
        # Fill missings
        x.fillna(0, inplace=True)
        # Feature Selection
        non_selected = [col for col in x.columns if col not in SELECTED_FEATURES]
        x.drop(non_selected, axis=1, inplace=True)
        # Normilize
        x = pd.DataFrame(self.scaler.fit_transform(x.astype(np.float64)), columns=[col for col in self.features if col in SELECTED_FEATURES])
        return x

    def fit(self, x, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return self.features
