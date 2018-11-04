from IPython.display import display

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

from pipelines import Pipeline
from utils import camelcase_to_underscore, save_model


df = pd.read_csv('input/train_V2.csv', nrows=10000)
df.columns = [camelcase_to_underscore(col) for col in df.columns]
display(df.head(), df.shape, list(df.columns))
df.drop(df[df['win_place_perc'].isnull()].index, inplace=True)

id_features = ['id', 'group_id', 'match_id']
categorical_features = ['match_type', ]
target_feature = 'win_place_perc'
base_features = [col for col in df.columns if col not in id_features + categorical_features + [target_feature]]

kfold = KFold(n_splits=2)
kfold.get_n_splits(df)
log = []
for train_index, valid_index in kfold.split(df):
    step = dict()
    pipeline = Pipeline(
        id_columns=id_features,
        numerical_columns=base_features,
        categorical_columns=categorical_features,
        target_column=target_feature,
    )
    x_train = pipeline.fit_transform(df.loc[train_index, :])
    y_train = df.loc[train_index, target_feature]

    print('Fitting ...')
    #     model = Ridge()
    model = RandomForestRegressor(
        n_jobs=-1,
        n_estimators=20,
        criterion='mae',
        max_depth=5,
    )
    model.fit(x_train, y_train)
    step['train_score'] = mean_absolute_error(y_train, model.predict(x_train))
    #     del x_train, y_train

    x_valid = pipeline.transform(df.loc[valid_index, :])
    y_valid = df.loc[valid_index, target_feature]

    step['valid_score'] = mean_absolute_error(y_valid, model.predict(x_valid))
    step['model'] = model
    step['pipeline'] = pipeline
    step['train_index'] = train_index
    step['valid_index'] = valid_index
    try:
        save_model(step)
    except Exception:
        print("Warning: Couldn't save the model")
    print(step['train_score'], step['valid_score'])
    log.append(step)
    break