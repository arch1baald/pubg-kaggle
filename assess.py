import gc
import os

from pipelines import Pipeline
from utils import kfold_with_respect_to_groups, save_model, Timer, scores_postprocessing


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
            continue
        os.remove(step['path'])
        if verbose == 2:
            print('Removed:', step['path'])
    return log
