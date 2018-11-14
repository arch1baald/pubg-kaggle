import gc
import os

import numpy as np

from pipelines import Pipeline
from utils import (
    kfold_with_respect_to_groups, save_model, Timer, postprocessing, load_data, split_columns_by_types
)


def assess(model, df, columns, metrics, n_splits=5, early_stopping_rounds=20, verbose=0):
    """
    k-fold cross-validation

    Checkpoints saving strategy ...
    :param model: sklearn-like object
    :param df: DataFrame with X and y
    :param columns: column names splited by types like utils.split_columns_by_types
    :param metrics: sklearn.metrics like function
    :param n_splits: the number of folds
    :param early_stopping_rounds: LightGBM param
    :param verbose: 0 - no logs, 1 - info, 2 - debug
    :return: iterations log
    """
    if n_splits == 1:
        total_rows = df.shape[0]
        train_size = int(0.95 * total_rows)
        splits = [(df.index[:train_size], df.index[train_size:])]
    else:
        splits = kfold_with_respect_to_groups(df, n_splits=n_splits)
    log = []
    for train_index, valid_index in splits:
        print('\n---------------------------')
        with Timer('Data Preparation:', verbose):
            pipeline = Pipeline(**columns, verbose=verbose)
            x_train = pipeline.fit_transform(df.loc[train_index, :])
            y_train = df.loc[train_index, columns['target']]
            x_valid = pipeline.transform(df.loc[valid_index, :])
            y_valid = df.loc[valid_index, columns['target']]

        with Timer('Fitting:', verbose):
            model.fit(
                x_train, y_train,
                eval_set=[(x_valid, y_valid)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=-1 if verbose != 2 else 1,
            )

        # with Timer('Postprocessing:', verbose):
        #     pred_train = scores_postprocessing(
        #         df=df.loc[train_index, :],
        #         predicted=model.predict(x_train),
        #         columns=columns,
        #         is_test=False,
        #     )[columns['target']]
        #     pred_valid = scores_postprocessing(
        #         df=df.loc[valid_index, :],
        #         predicted=model.predict(x_valid),
        #         columns=columns,
        #         is_test=False,
        #     )[columns['target']]
        pred_train, pred_valid = model.predict(x_train), model.predict(x_valid)

        with Timer('Saving:', verbose):
            train_score = metrics(y_train, pred_train)
            valid_score = metrics(y_valid, pred_valid)
            step = dict(
                model=model,
                pipeline=pipeline,
                train_score=train_score,
                valid_score=valid_score,
                not_adj_train_score=metrics(y_train, model.predict(x_train)),
                not_adj_valid_score=metrics(y_valid, model.predict(x_valid)),
                train_index=train_index,
                valid_index=valid_index,
                path=None,
                cached=False,
            )
            try:
                step = save_model(step)
            except Exception:
                if verbose == 1:
                    print("Warning: Couldn't save the model")
            log.append(step)
            gc.collect()

        if verbose == 1:
            print(step['train_score'], step['valid_score'])
        print('---------------------------\n')

    if verbose == 1:
        print('Erasing cache ...')
    for idx, step in enumerate(sorted(log, key=lambda dct: dct['valid_score'], reverse=True)):
        if idx == 0:
            step['best'] = True
            continue
        step['best'] = False

        try:
            os.remove(step['path'])
            if verbose == 2:
                print('Removed:', step['abspath'])
        except Exception:
            if verbose == 2:
                print("Warning: Couldn't remove file:", step['abspath'])
    return log


def main():
    from lightgbm import LGBMModel
    from sklearn.metrics import mean_absolute_error

    from assess import assess

    df = load_data('train', '../input', sample_size=10000)
    columns = split_columns_by_types(df)
    df.drop(df[df['win_place_perc'].isnull()].index, inplace=True)
    model_params = dict(
        objective='regression',
        metric='mae',
        # n_estimators=20000,
        n_estimators=2000,
        num_leaves=31,
        learning_rate=0.05,
        bagging_fraction=0.7,
        bagging_seed=0,
        num_threads=4,
        colsample_bytree=0.7
    )

    assessment_log = assess(
        LGBMModel(**model_params),
        df,
        columns,
        metrics=mean_absolute_error,
        n_splits=1,
        early_stopping_rounds=200,
        # early_stopping_rounds=20000,
        verbose=1,
    )
    del df

    best_model = [step for step in assessment_log if step['best']].pop()
    df_test = load_data('test', '../input')
    pipeline = best_model['pipeline']
    model = best_model['model']
    x_test = pipeline.transform(df_test)
    pred_test = model.predict(x_test)
    del df_test, x_test

    df_sub = load_data('sub', '../input', normilize_names=False)
    df_sub['winPlacePerc'] = pred_test
    df_sub_adjusted = postprocessing(pred_test, '../input')
    df_sub.to_csv('submission.csv', index=False)
    df_sub_adjusted.to_csv('submission_adjusted.csv', index=False)
    print(np.corrcoef(df_sub['winPlacePerc'], df_sub_adjusted['winPlacePerc']))
