import os
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dsap.datasets import build_dataset
from dsap.models import build_model
from dsap.utils import misc
from dsap.utils.config import load_config
from dsap.training import train_utils


def get_edges(model, dataset, params):
    no_gpu = params.get('no_gpu', False)
    batch_size = params['training']['batch_size']
    collate_fn = params.get('collate_fn', None)
    num_workers = params['training'].get('num_data_workers', 0)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=collate_fn, num_workers=num_workers)
    all_results = defaultdict(list)
    for batch_ind, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if not no_gpu:
            batch = train_utils.batch2gpu(batch)
        inputs = batch['inputs']
        labels = batch['labels']
        with torch.no_grad():
            edge_dict = model.predict_edges(inputs)
        for key, val in edge_dict.items():
            if key == 'dynamic_offsets':
                if key not in all_results:
                    all_results[key] = defaultdict(lambda: defaultdict(list))
                for recv, recv_dict in val.items():
                    for send, wts in recv_dict.items():
                        all_results[key][recv][send].append(wts.cpu().numpy())
            else:
                all_results[key].append(val.cpu().numpy())
            
    final_result = {}
    for key, val in all_results.items():
        if key == 'dynamic_offsets':
            final_dict = {}
            for recv, recv_dict in val.items():
                final_send_dict = {}
                for send, wt_list in recv_dict.items():
                    final_send_dict[send] = np.concatenate(wt_list, axis=0)
                final_dict[recv] = final_send_dict
            final_result[key] = final_dict
        else:
            final_result[key] = np.concatenate(val, axis=0)
    return final_result


if __name__ == '__main__':
    params = load_config([
        ['--out_name', {}]
    ])
    misc.seed(params['seed'])
    data = build_dataset(params, test=True)
    model = build_model(params)
    model.eval()
    working_dir = params['working_dir']
    for split, dataset in data.items():
        results = get_edges(model, dataset, params)
        for key, val in results.items():
            if params['out_name'] is not None:
                path = os.path.join(working_dir, '%s_%s_%s.npy'%(params['out_name'], key, split))
            else:
                path = os.path.join(working_dir, '%s_%s.npy'%(key, split))
            print("SAVING TO: ",path)
            np.save(path, val)
