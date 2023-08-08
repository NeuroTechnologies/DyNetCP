from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from . import model_utils


class Encoder(nn.Module):
    def __init__(self, embedder, params):
        super().__init__()
        num_vars = params['num_vars']
        self.num_vars = num_vars
        self.neuron_ids = params['neuron_ids']
        print("NEURON IDS: ",self.neuron_ids)
    
        self.embedder = embedder
        edge_file = params['model']['edge_path']
        edge_ind = params['model'].get('edge_ind')

        self.use_self = params['model']['encoder'].get('use_self')
        self.no_self = params['model']['encoder'].get('no_self')
        self.positional_encoding_size = params['model']['encoder'].get('positional_encoding_size', 0)
        init_gain = params['model']['encoder'].get('init_gain')
        self.pos_enc = None
        if self.positional_encoding_size > 0:
            self.use_positional_encoding = True
            max_len = params['data'].get('max_time', 0)
            if max_len > 0:
                self.pos_enc = nn.Parameter(
                    model_utils.get_positional_encoding(self.positional_encoding_size, max_len),
                    requires_grad = False,
                )
        else:
            self.use_positional_encoding = False
        if not self.no_self:
            raise ValueError('no_self must be enabled for sync model')
        self.edges = np.load(edge_file)
        if len(self.edges.shape) == 1:
            self.edges = self.edges[None, :]
        if edge_ind is not None:
            n1, n2 = self.edges[edge_ind]
            self.edges = np.array([[n1, n2],[n2, n1]])
        # Right now, removing self edges (probably not needed; won't work for delay 0)
        edges = []
        for i in range(len(self.edges)):
            if self.edges[i,0] != self.edges[i,1]: 
                edges.append([self.edges[i,0], self.edges[i,1]])
        self.edges = np.array(edges)
            
        print("FINAL NUM EDGES: ",len(self.edges))

        self.orig_send_edges = list(self.edges[:, 0])
        self.orig_recv_edges = list(self.edges[:, 1])
        self.used_neurons = sorted(list(np.unique(self.edges)))
        neuron_id_dict = {int(id):i for i,id in enumerate(self.neuron_ids)}
        self.used_neuron_idxs = [neuron_id_dict[id] for id in self.used_neurons]

        val_dict = {val:i for i,val in enumerate(self.used_neurons)}
        convert = np.vectorize(val_dict.get)
        self.val_dict = val_dict
        self.send_edges = convert(self.orig_send_edges)
        self.recv_edges = convert(self.orig_recv_edges)

        if embedder is not None:
            inp_size = params['model']['embedder']['embedding_size']
        else:
            raise NotImplementedError()
        self.node_embed_size = params['model']['encoder']['node_embed_size']

        self.num_edge_types = params['num_edge_types']
        self.send_recv_num_layers = params['model']['encoder'].get('send_recv_num_layers', 1)

        send_layers = []
        recv_layers = []
        for l_count in range(self.send_recv_num_layers-1):
            send_layers.append(nn.Linear(inp_size, inp_size))
            send_layers.append(nn.ReLU(inplace=True))
            recv_layers.append(nn.Linear(inp_size, inp_size))
            recv_layers.append(nn.ReLU(inplace=True))
        
        
        send_layers.append(nn.Linear(inp_size, self.node_embed_size))
        send_layers.append(nn.ReLU(inplace=True))
        recv_layers.append(nn.Linear(inp_size, self.node_embed_size))
        recv_layers.append(nn.ReLU(inplace=True))
    
    
        self.send_fc_out = nn.Sequential(*send_layers)
        self.recv_fc_out = nn.Sequential(*recv_layers)
        feat_size = 2*self.node_embed_size
        feat_size += 2*self.positional_encoding_size
        out_size = self.num_edge_types + 1
        inp_feat_size = feat_size
        
        layers = [nn.Linear(inp_feat_size, feat_size), nn.ReLU(inplace=True)]
        c_layers = params['model']['encoder'].get('num_combine_layers', 2)
        for _ in range(c_layers - 2):
            layers.append(nn.Linear(feat_size, feat_size))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(feat_size, out_size))
        self.combine = nn.Sequential(*layers)
        final_layer = self.combine[-1]
        final_layer.bias.data.fill_(0.0)
        if init_gain:
            final_layer.weight.data *= init_gain
        

    def get_edges(self):
        final_edges = [(send, recv) for send, recv in self.edges]
        return final_edges

    def forward(self, input_dict):
        inputs = input_dict['spikes']
        b, t, _, _ = inputs.shape
        inputs = inputs[:, :, self.used_neuron_idxs]
        if self.pos_enc is None:
            pos_enc = model_utils.get_positional_encoding(self.positional_encoding_size, t).to(device=inputs.device)
        else:
            pos_enc = self.pos_enc
        pos_enc = pos_enc.unsqueeze(0).expand(b, -1, -1)
        if self.embedder is not None:
            send_inputs = recv_inputs = self.embedder(inputs)
        else:
            send_inputs = recv_inputs = inputs

        send_embed = self.send_fc_out(send_inputs)
        send_embed = send_embed[:, :, self.send_edges]
        recv_embed = self.recv_fc_out(recv_inputs)
        recv_embed = recv_embed[:, :, self.recv_edges]
        pos_enc = pos_enc.unsqueeze(2).expand(-1, -1, send_embed.size(2), -1)[:, :send_embed.size(1)]
        send_embed = torch.cat([send_embed, pos_enc], dim=-1)
        recv_embed = torch.cat([recv_embed, pos_enc], dim=-1)
        # we need final order to be, e.g., [t-4, t-3, t-2, t-1, t]
        full_emb = torch.cat([send_embed[:, 1:], recv_embed[:, :-1]], dim=-1)
        result = self.combine(full_emb)
        
        final_result = defaultdict(dict)
        for idx, (send, recv) in enumerate(self.edges):
            final_result[recv][send] = result[:, :, idx]
        result_dict = {'dynamic_edge_weights': final_result}
        return result_dict

