import re
import bz2
import pickle
from datetime import datetime


def camelcase_to_underscore(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def save_model(pipeline_model_index_score):
    current_datetime = datetime.now().strftime('%d.%m.%Y-%H.%M.%S')
    str_valid_score = '{0:.5f}'.format(pipeline_model_index_score['valid_score'])
    name = f'valid_score_{str_valid_score}__{current_datetime}'
    path = f'models/{name}.pkl.bz2'
    with bz2.BZ2File(path, 'w') as fout:
        pickle.dump(pipeline_model_index_score, fout)


def load_model(path):
    with bz2.BZ2File(path, 'r') as fin:
        return pickle.load(fin)


def predict_from_file(df, path):
    model = load_model(path)
    x = model['pipeline'].transform(df)
    return model['model'].predict(x)
