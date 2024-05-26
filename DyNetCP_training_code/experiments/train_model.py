from dsap.utils.config import load_config
from dsap.models import build_model
from dsap.datasets import build_dataset
from dsap.training import train
from dsap.training import train_utils
from dsap.utils import misc


if __name__ == '__main__':
    params = load_config()

    misc.seed(params['seed'])
    misc.copy_config(params)
    datasets = build_dataset(params)
    model = build_model(params)
    with train_utils.build_writers(
            params['working_dir'], params['data']['data_splits']) as writers:
        train.train(model, datasets, params, writers)