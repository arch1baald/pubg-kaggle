import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


class SimpleFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Based on https://www.kaggle.com/deffro/eda-is-fun
    """
    def __init__(self):
        self.created_features = None
    
    def transform(self, df):
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
            assert self.created_features == list(df_features.columns)
        return df_features

    def fit(self, x, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        return self.created_features


class GroupAggregatedFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Based on https://www.kaggle.com/anycode/simple-nn-baseline-4
    """
    def __init__(self, features):
        self.created_features = None
        self.features = features

    def transform(self, df):
        df_features = []
        # Aggregate by Group
        for agg_type in ('mean', 'max', 'min'):
            df_aggregated = df.groupby(['match_id', 'group_id'], as_index=False)[self.features].agg(agg_type)
            df_aggregated = self.restore_row_order(df, df_aggregated, on=['match_id', 'group_id'])
            agg_column_names = {col: f'{agg_type}_group_{col}' for col in self.features}
            df_aggregated.rename(columns=agg_column_names, inplace=True)

            # Rank Groups by Match
            columns_to_select = list(agg_column_names.values())
            # Anyway deletes match_id
            df_ranked = df_aggregated.groupby('match_id', as_index=False)[columns_to_select].rank(pct=True)
            ranked_column_names = {col: f'rank_{col}' for col in columns_to_select}
            df_ranked.rename(columns=ranked_column_names, inplace=True)
            # Unsafe merge because of rank, which deletes match_id
            df_aggregated_ranked = pd.concat([df_aggregated, df_ranked], axis=1)
            df_features.append(df_aggregated_ranked)
            del df_aggregated, df_ranked
        df_features = pd.concat(df_features, axis=1)

        if self.created_features is None:
            self.created_features = list(df_features.columns)
        else:
            assert self.created_features == list(df_features.columns)
        return df_features

    def fit(self, x, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return self.created_features

    def restore_row_order(self, df, df_aggregated, on):
        """
        Восстановление индекса, FeatureUnion просто стакает колонки,
        поэтому результаты надо приводить к индексу в исходном датафрейме.
        :param df:
        :param df_aggregated:
        :param on:
        :return:
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
