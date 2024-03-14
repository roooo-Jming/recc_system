import torch.nn as nn
import torch


class SequencePoolingLayer(nn.Module):
    def __init__(self, mode='mean', support_mask=False, device='cpu'):
        super(SequencePoolingLayer, self).__init__()
        if mode not in ['sum', 'mean', 'max']:
            raise ValueError('parameter mode should in [sum,mean,max]')
        self.support_mask = support_mask
        self.device = device

    def _sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()

        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask.type(dtype)
        return mask
