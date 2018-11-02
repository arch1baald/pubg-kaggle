import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.created_features = None
    
    def transform(self, df):
        df_features = pd.DataFrame()
        df_features['total_distance'] = df['ride_distance'] + df['walk_distance'] + df['swim_distance']
        
        if self.created_features is None:
            self.created_features = list(df_features.columns)
        else:
            assert self.created_features == list(df_features.columns)
        return df_features

    def fit(self, x, y=None, **fit_params):
        return self
    
    def get_feature_names(self):
        return self.created_features
