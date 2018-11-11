from sklearn.base import BaseEstimator, TransformerMixin

from features import FeatureGenerator
from preprocessing import Preprocessor


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
        if self.verbose == 2:
            print('Transforming ...')
        self.feature_generator = FeatureGenerator(
            id=self.id,
            numeric=self.numeric,
            categorical=self.categorical,
            target=self.target,
        )
        df_features = self.feature_generator.fit_transform(df)

        self.preprocessor = Preprocessor(
            id=self.id,
            numeric=self.numeric,
            categorical=self.categorical,
            target=self.target,
        )
        x = self.preprocessor.fit_transform(df_features)
        return x

    def transform(self, df):
        if self.verbose == 2:
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

