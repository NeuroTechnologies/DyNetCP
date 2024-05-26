import os

import numpy as np
import torch
import h5py

from dsap.utils.config import load_config
from dsap.models import build_model
from dsap.datasets import build_dataset
from dsap.utils import misc


if __name__ == '__main__':
    params = load_config([
        ['--std_threshold', {'type':float}],
        ['--name', {'default':'thresholded_edges'}],
        ['--both_directions', {'action':'store_true'}],
        ['--correction_model_path', {}],
        ['--use_abs', {'action':'store_true'}],
        ['--spikes_path', {}],
        ['--joint_spike_threshold', {'type':float}],
        ['--wide_peak_thresh', {'type':float}],
        ['--outsides_range', {'type':int, 'default':50}],
    ])
    misc.seed(params['seed'])
    datasets = build_dataset(params)
    model = build_model(params)
    n_ids = params['neuron_ids']
    wide_peak_thresh = params['wide_peak_thresh']
    static_wts = model.get_static_weights().squeeze(0).cpu().numpy()
    final_edges = []
    std_threshold = params['std_threshold']
    outsides_range = params['outsides_range']
    if params['correction_model_path'] is not None:
        all_wts = torch.load(params['correction_model_path'])
        corrected_wts = all_wts['decoder']['model.weight'].permute(1,0,2)
        corrected_wts = torch.flip(corrected_wts, (-1,)).cpu().numpy()
        use_correction = True
    else:
        use_correction = False
    num_neurons = static_wts.shape[0]
    spike_thresh = params['joint_spike_threshold']
    if spike_thresh is not None:
        spike_rates = []
        with h5py.File(params['spikes_path'], 'r') as fin:
            num_trials = fin['train'].shape[0]
            num_bins = fin['train'].shape[1]
            for idx in range(num_neurons):
                spike_rates.append(fin['train'][:, :, idx].mean())
    for send_idx in range(static_wts.shape[0]):
        for recv_idx in range(send_idx+1, static_wts.shape[1]):
            if spike_thresh is not None:
                spike_val = spike_rates[send_idx]*spike_rates[recv_idx]*num_trials*num_bins*10
                if spike_val < spike_thresh:
                    continue
            forward_wts = static_wts[send_idx, recv_idx]
            back_wts = static_wts[recv_idx, send_idx]
            if use_correction:
                c_forward_wts = corrected_wts[send_idx, recv_idx]
                c_back_wts = corrected_wts[recv_idx, send_idx]
                forward_wts = forward_wts - c_forward_wts
                back_wts = back_wts - c_back_wts

            outsides = np.concatenate([forward_wts[-outsides_range:], back_wts[-outsides_range:]])
            std = outsides.std()
            mean = outsides.mean()
            if params['use_abs']:
                peak_forward = np.abs(forward_wts[:11]).max()
                peak_reverse = np.abs(back_wts[:11]).max()
                
                reverse_idx = np.abs(back_wts[:11]).argmax()
                forward_idx = np.abs(forward_wts[:11]).argmax()
            else:
                peak_forward = (forward_wts[:11]).max()
                peak_reverse = (back_wts[:11]).max()
                reverse_idx = back_wts[:11].argmax()
                forward_idx = forward_wts[:11].argmax()
            if wide_peak_thresh is not None:
                all_wts = np.concatenate([np.flip(forward_wts[1:], (-1,)), np.array([max(forward_wts[0], back_wts[0])]), back_wts[1:]])
                abs_wts = np.abs(all_wts)
                local_maxima = (abs_wts[1:-1] > abs_wts[2:])*(abs_wts[1:-1] > abs_wts[:-2])
                local_max_vals = abs_wts[1:-1][local_maxima]
            if forward_idx == 0 and reverse_idx == 0 and peak_forward > std_threshold*std:
                if wide_peak_thresh is None or sum(local_max_vals > peak_forward*wide_peak_thresh) == 1:
                    final_edges.append([int(n_ids[send_idx]), int(n_ids[recv_idx])])
                    if params['both_directions']:
                        final_edges.append([int(n_ids[recv_idx]), int(n_ids[send_idx])])
            elif peak_forward > peak_reverse and peak_forward > std_threshold*std:
                if wide_peak_thresh is None or sum(local_max_vals > peak_forward*wide_peak_thresh) == 1:
                    final_edges.append([int(n_ids[send_idx]), int(n_ids[recv_idx])])
                    if params['both_directions']:
                        final_edges.append([int(n_ids[recv_idx]), int(n_ids[send_idx])])
            elif peak_forward < peak_reverse and peak_reverse > std_threshold*std:
                if wide_peak_thresh is None or sum(local_max_vals > peak_reverse*wide_peak_thresh) == 1:
                    final_edges.append([int(n_ids[recv_idx]), int(n_ids[send_idx])])
                    if params['both_directions']:
                        final_edges.append([int(n_ids[send_idx]), int(n_ids[recv_idx])])

    final_edges = np.array(final_edges).astype(int)
    print("FINAL EDGES: ",final_edges)
    print("NUM EDGES: ",len(final_edges))
    print("THRESHOLD: ",params['std_threshold'])
    out_path = os.path.join(params['working_dir'], '%s.npy'%params['name'])
    print("OUT PATH: ",out_path)
    np.save(out_path, final_edges)

    out_count_path = os.path.join(params['working_dir'], '%s_len.txt'%params['name'])
    with open(out_count_path, 'w') as fout:
        fout.write('%d'%len(final_edges))

