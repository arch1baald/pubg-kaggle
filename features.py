import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


class SimpleFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    https://www.kaggle.com/deffro/eda-is-fun

    """
    def __init__(self):
        self.created_features = None
    
    def transform(self, x):
        df_features = pd.DataFrame()
        df_features['players_joined'] = x.groupby('match_id')['match_id'].transform('count')
        df_features['total_distance'] = x['ride_distance'] + x['walk_distance'] + x['swim_distance']
        df_features['kills_norm'] = x['kills'] * ((100 - df_features['players_joined']) / 100 + 1)
        df_features['damage_dealt_norm'] = x['damage_dealt'] * ((100 - df_features['players_joined']) / 100 + 1)
        df_features['heals_and_boosts'] = x['heals'] + x['boosts']
        df_features['total_distance'] = x['walk_distance'] + x['ride_distance'] + x['swim_distance']
        df_features['boosts_per_walk_distance'] = x['boosts'] / (x['walk_distance'] + 1)
        df_features['boosts_per_walk_distance'].fillna(0, inplace=True)
        df_features['heals_per_walk_distance'] = x['heals'] / (x['walk_distance'] + 1)
        df_features['heals_per_walk_distance'].fillna(0, inplace=True)
        df_features['heals_and_boosts_per_walk_distance'] = df_features['heals_and_boosts'] / (x['walk_distance'] + 1)
        df_features['heals_and_boosts_per_walk_distance'].fillna(0, inplace=True)
        df_features['kills_per_walk_distance'] = x['kills'] / (x['walk_distance'] + 1)
        df_features['kills_per_walk_distance'].fillna(0, inplace=True)
        df_features['team'] = [1 if i > 50 else 2 if (bool(i > 25) & bool(i <= 50)) else 4 for i in x['num_groups']]
        
        if self.created_features is None:
            self.created_features = list(df_features.columns)
        else:
            assert self.created_features == list(df_features.columns)
        return df_features

    def fit(self, x, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        return self.created_features


class ColumnsSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.created_features = None

    def transform(self, df_x):
        df_selected = df_x[self.columns].copy()
        self.created_features = list(df_selected)
        return df_selected

    def fit(self, x, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return self.created_features
