from sklearn.base import BaseEstimator, TransformerMixin

from features import FeatureGenerator
from preprocessing import Preprocessor


class NotFittedError(Exception):
    pass


class Pipeline(BaseEstimator, TransformerMixin):
    """
    """
    def __init__(self, numerical_columns, id_columns=None, target_column=None, categorical_columns=None):
        self.created_features = None
        self.id_columns = id_columns
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns

        self.feature_generator = None
        self.preprocessor = None

    def fit_transform(self, df, y=None, **fit_params):
        print('Transforming ...')
        self.feature_generator = FeatureGenerator(
            id_columns=self.id_columns,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column,
        )
        df_features = self.feature_generator.fit_transform(df)

        self.preprocessor = Preprocessor(
            id_columns=self.id_columns,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column,
        )
        x = self.preprocessor.fit_transform(df_features)
        return x

    def transform(self, df):
        print('Transforming ...')
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

