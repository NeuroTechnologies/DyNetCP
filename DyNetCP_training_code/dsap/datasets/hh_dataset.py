import torch
from torch.utils.data import Dataset
import os
import numpy as np
from tqdm import tqdm
import h5py

class HHDataset(Dataset):
    def __init__(self, split, params, test=False):
        self.split = split
        self.trial_len = params['data'].get('trial_len')
        self.use_jitter_correction = params['data'].get('use_jitter_correction')
        self.only_jitter_correction = params['data'].get('only_jitter_correction')
        self.jitter_correction_window = params['data'].get('jitter_correction_window')
        self.jitter_correction_type = params['data'].get('jitter_correction_type')
        self.is_small_data = params['data'].get('is_small_data')
        self.pairs_file = params['data'].get('pairs_file')
        self.pair_idx = params['data'].get('pair_idx')
        self.max_time = params['data'].get('max_time')
        assert self.jitter_correction_type in [None, 'trial', 'trial_exact', 'uniform_window']
        data_dir = params['data'].get('data_dir')
        if self.is_small_data:
            self.max_time = 5400000
            self.num_neurons = 20
            self.spikes = []
            avg_spike_rates = []
            for i in tqdm(range(self.num_neurons)):
                path = os.path.join(data_dir, 'cell%d.txt'%i)
                times = np.loadtxt(path)
                times = times[times > 0]
                split_idx = np.searchsorted(times, 4320000)
                if self.split == 'train':
                    times = np.floor(times[:split_idx]).astype(int)
                elif self.split == 'val':
                    split_idx_2 = np.searchsorted(times, 5400000)
                    times = np.floor(times[split_idx:split_idx_2] - 4320000).astype(int)
                else:
                    raise ValueError('split not recognized: ', self.split)
                self.spikes.append(times)
                if self.split == 'train':
                    avg_spike_rates.append(len(times)/4320000)
        else:
            prefixes = ['destexhe', 'erisir']
            counts = [800, 200]
            self.num_neurons = sum(counts)
            self.spikes = []
            avg_spike_rates = []
            if self.max_time is None:
                self.max_time = 10000000
            train_time = 0.8*self.max_time
            for prefix, count in zip(prefixes, counts):
                for idx in tqdm(range(count)):
                    path = os.path.join(data_dir, '%s%05d.txt'%(prefix, idx))
                    times = np.loadtxt(path)
                    split_idx = np.searchsorted(times, train_time)
                    if self.split == 'train':
                        times = np.floor(times[:split_idx]).astype(int)
                    elif self.split == 'val':
                        split_idx_2 = np.searchsorted(times, self.max_time)
                        times = np.floor(times[split_idx:split_idx_2] - train_time).astype(int)
                    else:
                        raise ValueError('split not recognized: ', self.split)
                    self.spikes.append(times)
                    if self.split == 'train':
                        avg_spike_rates.append(len(times)/8000000)
        if self.split == 'train':
            params['avg_spike_rates'] = torch.FloatTensor(avg_spike_rates)
        params['neuron_ids'] = [str(x) for x in list(range(self.num_neurons))]
        params['num_vars'] = self.num_neurons
        params['collate_fn'] = collate_fn
        if self.pairs_file is not None:
            raise NotImplementedError()
        if self.use_jitter_correction:
            self.jitter_data()
            if self.jitter_correction_type == 'trial_exact':
                self.jitter_h5_path = os.path.join(data_dir, 'jittered_spikes.h5')
                if self.split == 'train':
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'w')
                else:
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'a')
                if self.split not in self.jitter_h5:
                    if split == 'train':
                        n_len = 8000000
                    else:
                        n_len = 2000000
                    dset_shape = (n_len, 1000)
                    self.jitter_h5.create_dataset(self.split, shape=dset_shape)
                    for n in tqdm(range(1000)):
                        spikes = np.zeros(n_len)
                        spikes[self.spikes[n]] = 1
                        spikes = spikes.reshape((-1, 1000, 1)).transpose(1,0,2)
                        jittered = jitter(spikes, self.jitter_correction_window).transpose(1,0,2).flatten()
                        self.jitter_h5[self.split][:, n] = jittered
                self.jitter_h5.close()
                self.jitter_h5 = None

    def jitter_data(self):
        if self.jitter_correction_type == 'trial_exact':
            return
        self.jittered = []
        for n in range(self.num_neurons):
            final_times = []
            n_times = self.spikes[n]
            if self.jitter_correction_type == 'trial':
                trials = n_times // 1000
                trial_time = n_times % 1000
                for subset in range(0, 1000//25):
                    valid = trial_time // 25 == subset
                    current_trials = trials[valid]
                    current_tt = np.copy(trial_time[valid])
                    np.random.shuffle(current_tt)
                    new_times = current_trials*1000 + current_tt
                    final_times.append(new_times)
                final_times = np.concatenate(final_times)
                final_times = np.sort(final_times)
                self.jittered.append(final_times)
            elif self.jitter_correction_type == 'uniform_window':
                offset = np.random.randint(-self.jitter_correction_window, self.jitter_correction_window+1, size=n_times.shape)
                new_spikes = np.sort(n_times + offset)
                self.jittered.append(new_spikes)
            else:
                raise NotImplementedError()
    
    def __len__(self):
        if self.split == 'train':
            #return 8000 - 1
            return  int(0.8*self.max_time/1000) - 1
        else:
            #return 2000 - 1
            return int(0.2*self.max_time/1000) - 1

    def __getitem__(self, idx):
        if self.only_jitter_correction:
            result = {
                'inputs':{},
                'labels':{},
            }
        else:
            final_spikes = np.zeros((self.trial_len, self.num_neurons))
            t = idx*1000 + np.random.randint(0, 1000)
            for n_idx in range(self.num_neurons):
                n_times = self.spikes[n_idx]
                first_idx = np.searchsorted(n_times, t)
                second_idx = np.searchsorted(n_times, t+self.trial_len)
                for s_idx in range(first_idx, second_idx):
                    s_t = n_times[s_idx]
                    final_spikes[s_t-t, n_idx] = 1
            final_spikes = torch.from_numpy(final_spikes).float()
            result = {
                'inputs':{
                    'spikes': final_spikes.unsqueeze(-1),
                },
                'labels':{

                },

            }
        if self.use_jitter_correction:
            final_spikes = np.zeros((self.trial_len, self.num_neurons))
            if self.jitter_correction_type == 'trial_exact':
                if self.jitter_h5 is None:
                    self.jitter_h5 = h5py.File(self.jitter_h5_path, 'r')
                final_spikes = torch.from_numpy(
                    self.jitter_h5[self.split][t:t+self.trial_len]
                ).float()
                
            else:
                for n_idx in range(self.num_neurons):
                    n_times = self.jittered[n_idx]
                    first_idx = np.searchsorted(n_times, t)
                    second_idx = np.searchsorted(n_times, t+self.trial_len)
                    for s_idx in range(first_idx, second_idx):
                        s_t = n_times[s_idx]
                        final_spikes[s_t-t, n_idx] = 1
                final_spikes = torch.from_numpy(final_spikes).float()
            if self.only_jitter_correction:
                result['inputs']['spikes'] = final_spikes.unsqueeze(-1)
            else:
                result['inputs']['corrected_spikes'] = final_spikes.unsqueeze(-1)
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
    datasets = {split:HHDataset(split, params, test=test) for split in splits}
    return datasets


def jitter(data, l):
    """
    Jittering multidemntational logical data where 
    0 means no spikes in that time bin and 1 indicates a spike in that time bin.
    """
    #input shape should be [time, ntrials, num_neurons]
    if len(np.shape(data))>3:
        flag = 1
        sd = np.shape(data)
        data = np.reshape(data,(np.shape(data)[0],np.shape(data)[1],len(data.flatten())/(np.shape(data)[0]*np.shape(data)[1])), order='F')
    else:
        flag = 0
    psth = np.mean(data,axis=1)
    length = np.shape(data)[0]

    # pad to be divisible by l
    if np.mod(np.shape(data)[0],l):
        data[length:(length+np.mod(-np.shape(data)[0],l)),:,:] = 0
        psth[length:(length+np.mod(-np.shape(data)[0],l)),:]   = 0

    #dataj and psthj sums over the window (size l)
    #if np.shape(psth)[1]>1:
    #dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0))
    #psthj = np.squeeze(np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0))
    dataj = np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1],np.shape(data)[2]], order='F'), axis=0)
    psthj = np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l,np.shape(psth)[1]], order='F'), axis=0)
    #else:
    #    dataj = np.squeeze(np.sum(np.reshape(data,[l,np.shape(data)[0]//l,np.shape(data)[1]], order='F')))
    #    psthj = np.sum(np.reshape(psth,[l,np.shape(psth)[0]//l], order='F'))
    
    
    # this ensures that we didn't squeeze out the first dim (if time == l)
    if np.shape(data)[0] == l:
        dataj = np.reshape(dataj,[1,np.shape(dataj)[0],np.shape(dataj)[1]], order='F');
        psthj = np.reshape(psthj,[1,np.shape(psthj[0])], order='F');

    psthj = np.reshape(psthj,[np.shape(psthj)[0],1,np.shape(psthj)[1]], order='F')
    psthj[psthj==0] = 10e-10

    # corr is the number of spikes in each window (per trial, per coarse trial time)
    #    divided by the spike rate (averaged over trials) summed over the window
    corr = dataj/np.tile(psthj,[1, np.shape(dataj)[1], 1]);
    corr = np.reshape(corr,[1,np.shape(corr)[0],np.shape(corr)[1],np.shape(corr)[2]], order='F')
    corr = np.tile(corr,[l, 1, 1, 1])
    corr = np.reshape(corr,[np.shape(corr)[0]*np.shape(corr)[1],np.shape(corr)[2],np.shape(corr)[3]], order='F');

    psth = np.reshape(psth,[np.shape(psth)[0],1,np.shape(psth)[1]], order='F');
    # this final output multiplies corr by the spike rate at every point in time
    output = np.tile(psth,[1, np.shape(corr)[1], 1])*corr

    output = output[:length,:,:]
    return output