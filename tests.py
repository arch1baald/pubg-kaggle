def test_splits(df, splits):
    """
    To apply adjustment trick, players of one group in matches have to fall in the same fold.
    :param df: DataFrame
    :param splits:  [(train_idx, valid_idx), ..., (train_idx, valid_idx)]
    :return:
    """
    for train_idx, valid_idx in splits:
        count_intersections = (
            df.loc[valid_idx, 'group_id'].isin(df.loc[train_idx, 'group_id'])
            & df.loc[valid_idx, 'match_id'].isin(df.loc[train_idx, 'match_id'])
        ).sum()
        assert count_intersections == 0, f'Groups from single matches splited to train and test: {count_intersections}'
