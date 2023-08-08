import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import h5py
from .data_utils import jitter


class NeuroH5Dataset(Dataset):
    def __init__(self, split, params, test=False):
        self.split = split
        self.train_data_len = params['data'].get('train_data_len', 0)
        self.val_data_len = params['data'].get('val_data_len', 0)
        self.random_offset = params['data'].get('random_offset', 0)
        self.correction_type = params['data'].get('correction')
        self.neuron_subset = params['data'].get('neuron_subset')
        self.min_time = params['data'].get('min_time', 0)
        self.max_time = params['data'].get('max_time')
        self.use_jitter_correction = params['data'].get('use_jitter_correction')
        self.only_jitter_correction = params['data'].get('only_jitter_correction')
        self.jitter_correction_window = params['data'].get('jitter_correction_window')
        self.jitter_correction_type = params['data'].get('jitter_correction_type', 'trial')
        assert self.jitter_correction_type in ['trial', 'trial_exact']
        self.spikes_path = params['data']['spikes_path']
        self.spikes_h5 = None
        self.pairs_file = params['data'].get('pairs_file')
        self.pair_idx = params['data'].get('pair_idx')
        with h5py.File(self.spikes_path, 'r') as fin:
            self.spikes_shape = fin[self.split].shape
            if self.max_time is None or self.max_time > self.spikes_shape[1]:
                self.max_time = self.spikes_shape[1]
            if self.split == 'train':
                total_spikes = np.zeros(self.spikes_shape[2])
                complete_spikes = np.zeros((self.max_time-self.min_time, self.spikes_shape[2]))
                for idx in range(self.spikes_shape[0]):
                    if self.neuron_subset is None:
                        spikes = fin[self.split][idx, self.min_time:self.max_time].clip(0,1)
                    else:
                        spikes = fin[self.split][idx, self.min_time:self.max_time, self.neuron_subset].clip(0,1)
                    total_spikes += spikes.sum(0)
                    complete_spikes += spikes
                self.avg_spike_rates = total_spikes / (self.spikes_shape[0]*(self.max_time-self.min_time))
                params['avg_spike_rates'] = torch.FloatTensor(self.avg_spike_rates)
                dynamic_spr = complete_spikes / self.spikes_shape[0]
                params['avg_dyn_spike_rates'] = torch.FloatTensor(dynamic_spr)
        self.num_neurons = self.spikes_shape[2]
        params['neuron_ids'] = [str(x) for x in list(range(self.num_neurons))]
        params['num_vars'] = self.num_neurons
        params['collate_fn'] = collate_fn
        self.filter_pairs = params['data'].get('filter_pairs')
        self.filter_window = params['data'].get('filter_window')
        self.expand_train = params['data'].get('expand_train')
        
        if self.pairs_file is not None:
            if self.pairs_file.endswith('npy'):
                all_pairs = np.load(self.pairs_file)
            else:
                all_pairs = []
                with open(self.pairs_file, 'r') as fin:
                    for line in fin:
                        info = line.strip().split(',')
                        all_pairs.append([int(info[0]), int(info[1])])
            if self.pair_idx is not None:
                pair = all_pairs[self.pair_idx]
                params['num_vars'] = 2
                self.neuron_subset = np.sort(pair)
                params['neuron_ids'] = [params['neuron_ids'][x] for x in pair]
            else:
                used_neurons = np.unique(all_pairs)
                params['num_vars'] = len(used_neurons)
                self.neuron_subset = used_neurons
                params['neuron_ids'] = [params['neuron_ids'][x] for x in used_neurons]
            if self.split == 'train':
                if self.pair_idx is not None:
                    print("RUNNING ON PAIR: ",pair)
                else:
                    print("RUNNING WITH NEURONS: ",self.neuron_subset)
                self.avg_spike_rates = self.avg_spike_rates[self.neuron_subset]
                print("NEURON IDS: ",params['neuron_ids'])
                params['avg_spike_rates'] = params['avg_spike_rates'][self.neuron_subset]
                params['avg_dyn_spike_rates'] = params['avg_dyn_spike_rates'][:, self.neuron_subset]
        if self.use_jitter_correction:
            print("GETTING INITIAL SPIKE DATA")
            self.spikes_h5 = h5py.File(self.spikes_path, 'r')
            dset = self.spikes_h5[self.split]
            self.spike_inds = []
            t = self.spikes_shape[1]
            if self.neuron_subset is None:
                n_range = range(self.num_neurons)
            else:
                n_range = self.neuron_subset
            if self.jitter_correction_type == 'trial':
                self.spike_inds = []
                t = self.spikes_shape[1]
                
                for n in tqdm(n_range):
                        spikes = dset[:, :, n]
                        inds_for_n = []
                        self.spike_inds.append(inds_for_n)
                        for start_ind in np.arange(0, t, self.jitter_correction_window):
                            b_inds, sp_inds = spikes[:, start_ind:start_ind+self.jitter_correction_window].nonzero()
                            inds_for_n.append((b_inds, sp_inds))
            elif self.jitter_correction_type == 'trial_exact':
                self.jitter_h5_path = os.path.splitext(self.spikes_path)[0] + '_jittered.h5'
                if self.split == 'train':
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'w')
                else:
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'a')
                self.jitter_h5.create_dataset(self.split, shape=dset.shape)
                self.jitter_spikes = []
                for n in tqdm(n_range):
                    spikes = dset[:, :, n:n+1].transpose(1,0,2)
                    jittered = jitter(spikes, self.jitter_correction_window).transpose(1,0,2)
                    self.jitter_h5[self.split][:, :, n:n+1] = jittered
                self.jitter_h5.close()
                self.jitter_h5 = None 
        
    def jitter_data(self):
        if self.jitter_correction_type == 'trial_exact':
            return
        elif self.jitter_correction_type == 'trial':
            if self.neuron_subset is None:
                n_range = self.num_neurons
            else:
                n_range = len(self.neuron_subset)
            for n in range(n_range):
                for w in range(len(self.spike_inds[n])):
                    b_inds, sp_inds = self.spike_inds[n][w]
                    np.random.shuffle(sp_inds)
                    self.spike_inds[n][w] = (b_inds, sp_inds)
            
    def __len__(self):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return self.spikes_shape[0]

    def __getitem__(self, idx):
        if self.spikes_h5 is None:
            self.spikes_h5 = h5py.File(self.spikes_path, 'r')
        

        if self.neuron_subset is None:
            spikes = torch.from_numpy(self.spikes_h5[self.split][idx, self.min_time:self.max_time].clip(0,1)).unsqueeze(-1).float()
        else:
            spikes = torch.from_numpy(self.spikes_h5[self.split][idx, self.min_time:self.max_time, self.neuron_subset].clip(0,1)).unsqueeze(-1).float()
        size = len(spikes)
        if ((self.split == 'train' and self.train_data_len > 0) or (self.split != 'train' and self.val_data_len > 0)) and size > self.train_data_len:
            start_ind = np.random.randint(0, size - self.train_data_len)
            spikes = spikes[start_ind:start_ind+self.train_data_len]
        if self.split == 'train' and self.random_offset > 0:
            offset = np.random.randint(-self.random_offset, self.random_offset)
            if offset < 0:
                spikes = torch.cat([
                    spikes[-offset:],
                    torch.zeros(-offset, *spikes.shape[1:]),
                ], dim=0)
            elif offset > 0:
                spikes = torch.cat([
                    torch.zeros(offset, *spikes.shape[1:]),
                    spikes[:-offset],
                ], dim=0)
            
        result = {
            'inputs':{
                'spikes':spikes,
            },
            'labels':{

            },
        }
        if self.use_jitter_correction:
            if self.jitter_correction_type == 'trial':
                if self.neuron_subset is None:
                    n_size = self.num_neurons
                    jittered_spikes = np.zeros(self.spikes_shape[1:])
                else:
                    n_size = len(self.neuron_subset)
                    jittered_spikes = np.zeros((self.spikes_shape[1], n_size))
                for n in range(n_size):
                    spike_inds = self.spike_inds[n]
                    for start_ind, (b_inds, sp_inds) in zip(np.arange(0, self.spikes_shape[1], self.jitter_correction_window), spike_inds):
                        valid = b_inds == idx
                        jittered_spikes[sp_inds[valid]+start_ind, n] = 1
                jittered_spikes = torch.FloatTensor(jittered_spikes).unsqueeze(-1)
            elif self.jitter_correction_type == 'trial_exact':
                if self.jitter_h5 is None:
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'r')
                jittered_spikes = torch.from_numpy(self.jitter_h5[self.split][idx, self.min_time:self.max_time]).float().unsqueeze(-1)

            if self.only_jitter_correction:
                result['inputs']['spikes'] = jittered_spikes
            else:
                result['inputs']['corrected_spikes'] = jittered_spikes

        return result



def collate_fn(batch):
    inputs = [entry['inputs'] for entry in batch]
    labels = [entry['labels'] for entry in batch]

    result_inp = {}
    for key in inputs[0]:
        result_inp[key] = torch.stack([entry[key] for entry in inputs])
    result_label = {}
    for key in labels[0]:
        result_label[key] = torch.stack([entry[key] for entry in labels])
    return {'inputs': result_inp, 'labels': result_label}

def build_dataset(params, test=False):
    splits = params['data']['data_splits']
    datasets = {split:NeuroH5Dataset(split, params, test=test) for split in splits}
    return datasets