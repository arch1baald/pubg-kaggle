import pandas as pd
import numpy as np

from utils import camelcase_to_underscore
from pipelines import Pipeline


def asdf():
    df = pd.read_csv('input/train_V2.csv', nrows=10000)
    df.columns = [camelcase_to_underscore(col) for col in df.columns]
    df.drop(df[df['win_place_perc'].isnull()].index, inplace=True)

    id_features = ['id', 'group_id', 'match_id']
    categorical_features = ['match_type', ]
    target_feature = 'win_place_perc'
    base_features = [col for col in df.columns if col not in id_features + categorical_features + [target_feature]] 
    pipeline = Pipeline(
        id_columns=id_features, 
        numerical_columns=base_features,
        categorical_columns=categorical_features,
        target_column=target_feature,
    )
    return pipeline.fit_transform(df)
