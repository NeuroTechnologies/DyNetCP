import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from .data_utils import jitter

class DefaultNeuroDataset(Dataset):
    def __init__(self, split, params, test=False):
        self.test = test
        self.split = split
        self.train_data_len = params['data'].get('train_data_len', 0)
        self.val_data_len = params['data'].get('val_data_len', 0)
        self.random_offset = params['data'].get('random_offset', 0)
        self.correction_type = params['data'].get('correction')
        self.neuron_subset = params['data'].get('neuron_subset')
        self.min_time = params['data'].get('min_time', 0)
        self.max_time = params['data'].get('max_time')
        self.flip_all = params['data'].get('flip_all')
        self.flip_augment = params['data'].get('flip_augment')
        self.use_jitter_correction = params['data'].get('use_jitter_correction')
        self.jitter_correction_type = params['data'].get('jitter_correction_type', 'trial')
        self.only_jitter_correction = params['data'].get('only_jitter_correction')
        self.jitter_correction_window = params['data'].get('jitter_correction_window')
        spikes_path = params['data']['spikes_path']
        train_spikes, val_spikes, test_spikes = np.load(spikes_path, allow_pickle=True)
        if self.split == 'train':
            spikes = train_spikes
        elif self.split == 'val':
            spikes = val_spikes
        elif self.split == 'test':
            spikes = test_spikes
        self.spikes = torch.FloatTensor(spikes).clamp(0, 1)
        if self.max_time is None:
            self.max_time = self.spikes.size(1)
        self.spikes = self.spikes[:, self.min_time:self.max_time]
        if self.neuron_subset is not None:
            self.spikes = self.spikes[:, :, self.neuron_subset]
        self.num_neurons = self.spikes.shape[2]
        #TODO: should do this properly
        params['neuron_ids'] = [str(x) for x in list(range(self.num_neurons))]
        params['num_vars'] = self.num_neurons
        params['collate_fn'] = collate_fn
        if self.split == 'train':
            self.avg_spike_rates = self.spikes.view(-1, self.num_neurons).mean(dim=0)
            params['avg_spike_rates'] = self.avg_spike_rates
        self.filter_pairs = params['data'].get('filter_pairs')
        self.filter_window = params['data'].get('filter_window')
        if len(self.spikes.shape) == 3:
            self.spikes = self.spikes.unsqueeze(-1)
        self.expand_train = params['data'].get('expand_train')
        if self.split == 'train' and self.expand_train and self.train_data_len > 0:
            self.all_inds = []
            for ind in range(len(self.spikes)):
                t_ind = 0
                while t_ind < len(self.spikes[ind]):
                    self.all_inds.append((ind, t_ind))
                    t_ind += self.train_data_len
        else:
            self.expand_train = False
        if self.filter_pairs:
            graph_cache = {}
            max_active = 0
            total_active = 0
            count = 0
            self.masks = []
            self.node_inds = []
            self.graph_info = []
            print("FILTERING PAIRS...")
            for data_ind in tqdm(range(len(self.spikes))):
                current_node_inds = []
                self.node_inds.append(current_node_inds)
                current_graph_info = []
                self.graph_info.append(current_graph_info)
                current_masks = []
                for time_ind in range(self.spikes.size(1)):
                    if time_ind < self.filter_window:
                        window = self.spikes[data_ind, :time_ind+self.filter_window+1]
                    else:
                        window = self.spikes[data_ind, time_ind-self.filter_window:time_ind+self.filter_window+1]
                    has_spiked = (window.sum(dim=0).squeeze(-1) > 0).float()
                    current_masks.append(has_spiked)
                    current_node_inds.append(has_spiked.nonzero()[:, -1])
                    node_len = len(current_node_inds[-1])
                    if node_len in graph_cache:
                        graph_info = graph_cache[node_len]
                    else:
                        graph_info = get_graph_info(has_spiked, node_len)
                    current_graph_info.append(graph_info)
                    max_active = max(max_active, has_spiked.sum().item())
                    total_active += has_spiked.sum().item()
                    count += 1
                self.masks.append(torch.stack(current_masks))
            print("MAX ACTIVE: ",max_active)
            print("MEAN ACTIVE: ",total_active/count)

        if self.use_jitter_correction:
            if self.jitter_correction_type == 'trial_exact':
                self.jitter_spikes = []
                for n in tqdm(range(self.num_neurons)):
                    spikes = self.spikes[:, :, n:n+1, 0].numpy().transpose(1,0,2)
                    jittered = jitter(spikes, self.jitter_correction_window).transpose(1,0,2)
                    self.jitter_spikes.append(jittered)
                self.jitter_spikes = torch.FloatTensor(np.concatenate(self.jitter_spikes, axis=-1)).unsqueeze(-1)
            else:
                raise NotImplementedError('No implementation for jitter type: ', self.jitter_correction_type)


    def jitter_data(self):
        if self.jitter_correction_type == 'trial_exact':
            return
        else:
            raise NotImplementedError()
            
    def __len__(self):
        if self.expand_train:
            return len(self.all_inds)
        else:
            return len(self.spikes)

    def __getitem__(self, idx):
        if self.expand_train:
            ind, t_ind = self.all_inds[idx]
            start_ind = np.random.randint(t_ind, t_ind + self.train_data_len)
            spikes = self.spikes[ind, start_ind:start_ind + self.train_data_len]
            
            if len(spikes) < self.train_data_len:
                spikes = self.spikes[ind, -self.train_data_len:]
            if self.filter_pairs:
                masks = self.masks[idx][start_ind:start_ind+self.train_data_len]
                node_inds = self.node_inds[idx][start_ind:start_ind+self.train_data_len]
                graph_info = self.graph_info[idx][start_ind:start_ind+self.train_data_len]
                if len(spikes) < self.train_data_len:
                    masks = self.masks[idx][-self.train_data_len:]
                    node_inds = self.node_inds[idx][-self.train_data_len:]
                    graph_info = self.graph_info[idx][-self.train_data_len:]

        else:
            spikes = self.spikes[idx]
            size = len(spikes)
            if self.filter_pairs:
                masks = self.masks[idx]
                node_inds = self.node_inds[idx]
                graph_info = self.graph_info[idx]
            if ((self.split == 'train' and self.train_data_len > 0) or (self.split != 'train' and self.val_data_len > 0)) and size > self.train_data_len:
                start_ind = np.random.randint(0, size - self.train_data_len)
                spikes = spikes[start_ind:start_ind+self.train_data_len]
                if self.filter_pairs:
                    masks = masks[start_ind:start_ind+self.train_data_len]
                    node_inds = node_inds[start_ind:start_ind+self.train_data_len]
                    graph_info = graph_info[start_ind:start_ind+self.train_data_len]
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
        if self.flip_all:
            spikes = 1-spikes
        elif self.split == 'train' and not self.test and self.flip_augment and np.random.random() > 0.5:
            spikes = 1-spikes
        result = {
            'inputs':{
                'spikes':spikes,
            },
            'labels':{

            },
        }
        if self.filter_pairs:
            result['inputs']['masks'] = masks
            result['inputs']['node_inds'] = node_inds
            result['inputs']['graph_info'] = graph_info
        if self.correction_type is not None:
            if self.correction_type == 'shuffle':
                rand_inds = np.random.randint(0, high=len(self.spikes), size=self.num_neurons)
                corrected_spikes = self.spikes[rand_inds, :, range(self.num_neurons)]
                corrected_spikes = np.transpose(corrected_spikes, (1, 0, 2))
                
            elif self.correction_type == 'jitter':
                pass
            result['inputs']['corrected_spikes'] = torch.FloatTensor(corrected_spikes)
        if self.use_jitter_correction:
            if self.jitter_correction_type == 'trial_exact':
                jittered_spikes = self.jitter_spikes[idx]
            if self.only_jitter_correction:
                result['inputs']['spikes'] = jittered_spikes
            else:
                result['inputs']['corrected_spikes'] = jittered_spikes
        return result


def get_graph_info(masks, num_vars):
    if num_vars == 1:
        return None, None
    edges = torch.ones(num_vars, device=masks.device) - torch.eye(num_vars, device=masks.device)
    tmp = torch.where(edges)
    send_edges = tmp[0]
    recv_edges = tmp[1]
    tmp_inds = torch.tensor(list(range(num_vars)), device=masks.device, dtype=torch.long).unsqueeze_(1)
    return send_edges, recv_edges


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
    datasets = {split:DefaultNeuroDataset(split, params, test=test) for split in splits}
    return datasets