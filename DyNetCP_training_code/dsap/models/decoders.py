import math

import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.num_vars = params['num_vars']
        self.num_edge_types = params['num_edge_types'] + 1
        avg_rates = torch.clamp(params['avg_spike_rates'], min=1e-6)
        spike_offsets = -torch.log(1/avg_rates-1)
    
        self.no_static = params['model']['decoder']['no_static']
        self.neuron_ids = params['neuron_ids']
        init_static_refractory_wt = params['model']['decoder'].get('init_static_refractory_wt')
        init_uniform = params['model']['decoder'].get('init_uniform')
        init_random_uniform = params['model']['decoder'].get('init_random_uniform')
        self.sparse_dynamic = params['model'].get('sparse_dynamic')
        self.compute_l2 = params['model'].get('l2_coef', 0) > 0
        self.neuron_id_inds = {}
        self.faster = params['model'].get('faster')
        
        for ind, n_id in enumerate(self.neuron_ids):
            self.neuron_id_inds[n_id] = ind
        self.model = nn.Conv1d(self.num_vars, self.num_vars, self.num_edge_types, padding=self.num_edge_types-1)
        self.model.bias.data = spike_offsets
        if init_static_refractory_wt:
            self.model.weight.data[range(self.num_vars), range(self.num_vars), -2] = -5
        if init_uniform is not None:
            self.model.weight.data.fill_(init_uniform)
        if init_random_uniform is not None:
            self.model.weight.data.uniform_(-init_random_uniform, init_random_uniform)
            self.model.weight.data = self.model.weight.data[0:1].repeat(self.num_vars, 1, 1)

    def _mask_self_weights(self):
        if self.no_static:
            return
        self.model.weight.data[range(self.num_vars), range(self.num_vars), -1] = 0

    def get_static_edges(self):
        self._mask_self_weights()
        # Conv weight is [recv, send, num_edge_types]
        if self.no_static:
            return {'static_weights': torch.zeros(1, self.num_vars, self.num_vars, self.num_edge_types)}
        wts = self.model.weight.data
        wts = wts.permute(1,0,2)
        wts = torch.flip(wts, (-1,)).unsqueeze(0)
        result = {'static_weights': wts}
        
        return result

    def forward(self, input_dict, dyn_dict, use_corrected=False, ignore_pairs=None):
        #need to mask self-weights for delay 0 (otherwise problem too easy)
        self._mask_self_weights()
        if use_corrected:
            inputs = input_dict['corrected_spikes']
        else:
            inputs = input_dict['spikes']
        b, t, nvar, _ = inputs.shape
        inputs = inputs.squeeze(-1).permute(0,2,1)
        if self.no_static:
            self.model.weight.data.fill_(0)
        result = self.model(inputs)[:, :, 1:-(self.num_edge_types-1)]
        result = result.permute(0, 2, 1).unsqueeze(-1)
        other_vals = {}
        
        #dynamic part
        if dyn_dict is not None:
            if self.compute_l2:
                total_l2 = 0
            dynamic_offset = dyn_dict['dynamic_edge_weights']
            sliding_window_inp = F.unfold(inputs.unsqueeze(-1), (self.num_edge_types, 1),
                                        padding=(self.num_edge_types-1, 0))
            if self.num_edge_types > 1:
                sliding_window_inp = sliding_window_inp[:, :, 1:-1*(self.num_edge_types-1)]
            sliding_window_inp = sliding_window_inp.reshape(b, nvar, self.num_edge_types, t-1)
            sliding_window_inp = sliding_window_inp.permute(0, 3, 1, 2)
            for ind,n_id in enumerate(self.neuron_ids):
                if self.compute_l2:
                    if self.no_static:
                        total_model = torch.zeros(b, t-1, self.num_vars, self.num_edge_types, device=inputs.device)
                    else:
                        wt = self.model.weight
                        total_model = wt[ind].reshape(1, 1, self.num_vars, self.num_edge_types).expand(b, t-1, -1, -1).clone()
                tmp_dict = dynamic_offset.get(int(n_id), {})
                if len(tmp_dict) == 0:
                    continue
                all_send_wts = torch.stack(list(tmp_dict.values()), dim=2)
                all_idxs = [self.neuron_id_inds[str(id)] for id in tmp_dict.keys()]
                all_inp = sliding_window_inp[:, :, all_idxs]
                tmp_result = (all_send_wts*all_inp).reshape(b, t-1, -1).sum(-1)
                result[:, :, ind, 0] = result[:, :, ind, 0] + tmp_result
                if self.compute_l2:
                    total_model[:, :, all_idxs] = total_model[:, :, all_idxs] + all_send_wts
                
                if self.compute_l2:
                    total_l2 = total_l2 + torch.square(total_model).reshape(b, -1).mean(-1)
                
            if self.compute_l2:
                total_l2 = total_l2 / self.num_vars
        else:
            if self.compute_l2:
                wt = self.model.weight
                total_l2 = torch.square(wt).mean().reshape(1).expand(b)
        if self.compute_l2:
            other_vals['l2_reg'] = total_l2
        return result, other_vals




