import torch
from torch import nn

from .base_model import BaseModel

class DSAP(BaseModel):
    def __init__(self, encoder, decoder, params):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_only = params['model'].get('encoder_only')
        self.decoder_only = params['model'].get('decoder_only')
        self.num_edge_types = params['num_edge_types']
        self.loss_type = params['model'].get('loss_type', 'bcewithlogits')
        self.l2_coef = params['model'].get('l2_coef', 0)
        self.correction_only = params['model'].get('correction_only', False)
        if self.encoder_only:
            for param in self.decoder.parameters():
                param.requires_grad = False
        self.train_with_correction = params['model'].get('train_with_correction')
        self.val_with_correction = params['model'].get('val_with_correction')
        self.corrected_nll_coef = params['model'].get('corrected_nll_coef')
        if self.train_with_correction:
            print("TRAINING WITH CORRECTION: ",self.corrected_nll_coef)
            if self.val_with_correction:
                print("VAL WITH CORRECTION")
        if self.correction_only:
            print("CORRECTION ONLY")

    def forward(self, inputs, use_corrected=False):
        if self.decoder_only:
            encoder_dict = None
        else:
            encoder_dict = self.encoder(inputs)
        decoder_result, all_wts = self.decoder(inputs, encoder_dict, use_corrected=use_corrected)
        return encoder_dict, all_wts, decoder_result

    def loss(self, inputs, labels):
        spikes = inputs['spikes']

        if self.correction_only:
            corrected_spikes = inputs['corrected_spikes']
            target_corrected_spikes = corrected_spikes[:, 1:]
            dyn_dict, other_vals, corrected_preds = self(inputs, use_corrected=True)
            corrected_nll = self.nll(corrected_preds, target_corrected_spikes)
            loss_dict = {}
            loss_dict['corrected_nll'] = corrected_nll
            loss = self.corrected_nll_coef*corrected_nll
        else:
            dyn_dict, other_vals, preds = self(inputs)
            target_spikes = spikes[:, 1:]
            nll = self.nll(preds, target_spikes)
            loss_dict = {
                'nll': nll,
            }
            loss = nll
        
        if self.l2_coef > 0:
            l2_reg = other_vals['l2_reg']
            loss_dict['l2_reg'] = l2_reg
            loss = loss + self.l2_coef*l2_reg
        
        if self.train_with_correction and not self.correction_only:
            corrected_spikes = inputs['corrected_spikes']
            target_corrected_spikes = corrected_spikes[:, 1:]
            _, _, corrected_preds = self(inputs, use_corrected=True)
            corrected_nll = self.nll(corrected_preds, target_corrected_spikes)
            loss_dict['corrected_nll'] = corrected_nll
            if self.training or self.val_with_correction:
                loss = loss + self.corrected_nll_coef*corrected_nll
        loss_dict['loss'] = loss
        return loss_dict

    def predict_spike_probs(self, inputs, labels):
        _, _, pred_spikes = self(inputs)
        return torch.sigmoid(pred_spikes)

    def predict_edges(self, inputs):
        result = {}
        static_weight_dict = self.decoder.get_static_edges()
        for key, val in static_weight_dict.items():
            result[key] = val
        if not self.decoder_only:
            dyn_dict = self.encoder(inputs)
            dynamic_offset = dyn_dict['dynamic_edge_weights']
            dynamic_result = {}
            for recv, recv_dict in dynamic_offset.items():
                new_recv_dict = {}
                dynamic_result[recv] = new_recv_dict
                for send, wts in recv_dict.items():
                    wts = torch.cat([
                        torch.zeros(wts.size(0), 1, wts.size(2), device=wts.device),
                        wts,
                    ], dim=1 )
                    new_recv_dict[send] = torch.flip(wts, dims=(-1,))
            result['dynamic_offsets'] = dynamic_result
        return result

    def get_static_weights(self):
        static_weight_dict = self.decoder.get_static_edges()
        return static_weight_dict['static_weights']

    def nll(self, preds, target):
        if self.loss_type == 'bcewithlogits':
            loss = nn.BCEWithLogitsLoss(reduction='none')(preds, target)
        elif self.loss_type == 'bce':
            loss = nn.BCELoss(reduction='none')(preds, target)
        return loss.view(preds.size(0), -1).mean(dim=1)

    def get_dynamic_edges(self):
        return self.encoder.get_edges()

    def save(self, path):
        state_dict = {}
        if self.encoder is not None:
            state_dict['encoder'] = self.encoder.state_dict()
        state_dict['decoder'] = self.decoder.state_dict()
        torch.save(state_dict, path)
    
    def load(self, path):
        state_dict = torch.load(path, map_location='cpu')
        if 'encoder' in state_dict:
            self.encoder.load_state_dict(state_dict['encoder'], strict=False)
        self.decoder.load_state_dict(state_dict['decoder'], strict=False)