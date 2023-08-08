from . import default_neuro_dataset, neuro_h5_dataset, hh_dataset

def build_dataset(params, test=False):
    dataset_type = params['data'].get('dataset_type', 'default')
    if dataset_type == 'default':
        return default_neuro_dataset.build_dataset(params, test=test)
    elif dataset_type == 'h5':
        return neuro_h5_dataset.build_dataset(params, test=test)
    elif dataset_type == 'hh':
        return hh_dataset.build_dataset(params, test=test)
    else:
        raise ValueError('Dataset type not recognized: ',dataset_type)