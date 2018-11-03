import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


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
        # Drop columns
        x = df.drop(self.id_columns + [self.target_column] + self.categorical_columns, axis=1).copy()
        # Fill missings
        x.fillna(0, inplace=True)
        # Normilize
        self.scaler = MinMaxScaler()
        self.features = x.columns
        x = pd.DataFrame(self.scaler.fit_transform(x.astype(np.float64)), columns=self.features)
        return x


    def transform(self, df):
        # Drop columns
        x = df.drop(self.id_columns + [self.target_column] + self.categorical_columns, axis=1).copy()
        # Fill missings
        x.fillna(0, inplace=True)
        # Normilize
        x = pd.DataFrame(self.scaler.fit_transform(x.astype(np.float64)), columns=self.features)
        return x

    def fit(self, x, y=None, **fit_params):
        return self

    def get_feature_names(self):
        return self.features
