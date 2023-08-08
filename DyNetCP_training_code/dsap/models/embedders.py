import torch
from torch import nn
import numpy as np
import scipy.stats


class RNNEmbed(nn.Module):
    def __init__(self, params):
        super(RNNEmbed, self).__init__()
        self.num_vars = params['num_vars']
        self.embedding_size = params['model']['embedder']['embedding_size']
        self.neuron_ids = params['neuron_ids']
        self.shift_spikes = params['model']['embedder'].get('shift_spikes')
        self.use_fourier_encoding = params['model']['embedder'].get('use_fourier_encoding')
        embed_all = params['model'].get('embed_all')
        width = params['model']['embedder'].get('fourier_encoding_width')
        std = params['model']['embedder'].get('fourier_encoding_std')
        wts = torch.linspace(-2*std, 2*std, width)
        self.encoding_wts = nn.Parameter(wts, requires_grad=False)
        inp_size = 2*width
        edge_file = params['model'].get('edge_path')
        use_self = params['model']['encoder'].get('use_self')
        no_self = params['model']['encoder'].get('no_self')
        if not use_self and edge_file is not None and not embed_all:
            self.edges = np.load(edge_file)
            edge_ind = params['model'].get('edge_ind')
            if edge_ind is not None:
                self.edges = self.edges[edge_ind:edge_ind+1]
            if len(self.edges.shape) == 1:
                self.edges = self.edges[None, :]
            if no_self:
                edges = []
                for i in range(len(self.edges)):
                    if self.edges[i,0] != self.edges[i,1]: 
                        edges.append([self.edges[i,0], self.edges[i,1]])
                self.edges = np.array(edges)

            self.neuron_ids = np.unique(self.edges)
            self.neuron_ids.sort()
            self.neuron_ids = self.neuron_ids.astype(str)
        n_layers = params['model']['embedder']['num_layers']
        embedding_size = self.embedding_size
        self.embedders = nn.ModuleDict()
        for n_id in self.neuron_ids:
            self.embedders[n_id] = nn.LSTM(inp_size, embedding_size, n_layers,
                                            batch_first=True)
        
    
    def forward(self, spikes):
        if self.shift_spikes:
            spikes = 2*spikes-1
        if self.use_fourier_encoding:
            spikes = torch.cat([
                torch.cos(2*np.pi*self.encoding_wts*spikes),
                torch.sin(2*np.pi*self.encoding_wts*spikes),
            ], dim=-1)
        results = []
        
        for ind, (n_id, embedder) in enumerate(self.embedders.items()):
            inp_spikes = spikes[:, :, ind]
            outputs, state = embedder(inp_spikes)
            results.append(outputs)
        result =  torch.stack(results, dim=2)
        return result

