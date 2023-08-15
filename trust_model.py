
import torch
import math
import numpy as np

from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=80, batch_first=False):
        super(PositionalEncoding, self).__init__()
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            return self.pe[:, :x.size(1)]
        else:
            return self.pe[:x.size(0)]



def which_dtype(array):
    if type(array) == torch.Tensor:
        return array.dtype
    if np.issubdtype(array.dtype, np.integer):
        return torch.long
    elif np.issubdtype(array.dtype, np.floating):
        return torch.float32
    elif np.issubdtype(array.dtype, np.bool_):
        return torch.bool



class SimpleTransformerEncoder2(nn.Module):
    def __init__(self, n_feature=64, n_head=2, n_layers=3, n_hidden=1024, dropout=0.5, n_out=2):
        super().__init__()
        self.segment_embedding = nn.Embedding(8, n_feature)
        self.pos_encoder = PositionalEncoding(n_feature)
        self.special_word_embedding = nn.Embedding(8, n_feature)

        self.img_encoder = nn.Linear(512, n_feature)
        self.instance_conf_encoder = nn.Linear(n_feature, n_feature)
        self.conf_mask_encoder = nn.Embedding(1, n_feature)
        self.action_encoder = nn.Embedding(5, n_feature) # 0: AI, 1: Human, 3: [MASK]
        self.feedback_encoder = nn.Embedding(5, n_feature) # 0: AI == Human, 1: AI != Human, 2: unknown, 3: [MASK]

        self.model_type = 'Transformer'

        encoder_layers = TransformerEncoderLayer(n_feature, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.n_feature = n_feature

        self.linear = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(n_feature, n_feature),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(n_feature, n_out),
        )


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.img_encoder.weight.data.uniform_(-initrange, initrange)
        self.instance_conf_encoder.weight.data.uniform_(-initrange, initrange)

    def preprocess(self, middles, domain_confs, instance_confs, actions, feedbacks):
        middles = self.img_encoder(middles)


        _instance_confs_embed = self.instance_conf_encoder(instance_confs.expand(instance_confs.shape[0], instance_confs.shape[1], self.n_feature))
        _instance_confs_embed[(instance_confs == -100).squeeze(2)] = self.conf_mask_encoder(torch.zeros_like(instance_confs[instance_confs == -100], dtype=torch.long))

        instance_confs = _instance_confs_embed

        actions = self.action_encoder(actions)

        feedbacks = self.feedback_encoder(feedbacks)
        src = torch.stack((middles, instance_confs, actions, feedbacks), axis=0)

        return src

    def forward(self, middles, middles_mask, domain_confs, instance_confs, actions, feedbacks, **other_masks):
        # src.shape(sequence_length, batch_size, feature_number)
        # feedback:  0: AI == Human, 1: AI != Human, 2: Unknown, 3: [MASK]
        # action: 0: AI, 1: Human, 3: [MASK]
        src = self.preprocess(middles,domain_confs, instance_confs, actions, feedbacks)

        src = torch.sum(src, axis=0)

        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=middles_mask)
        output = self.linear(output)
        return output.squeeze(2)


class SimpleTransformerEncoder_Access(nn.Module):
    def __init__(self, n_feature=64, n_head=2, n_layers=3, n_hidden=1024, dropout=0.5, n_out=2):
        super().__init__()
        self.segment_embedding = nn.Embedding(8, n_feature)
        self.pos_encoder = PositionalEncoding(n_feature)
        self.special_word_embedding = nn.Embedding(8, n_feature)

        self.img_encoder = nn.Linear(512, n_feature)
        self.cue_encoder = nn.Embedding(5, n_feature)
        self.decision_encoder = nn.Embedding(5, n_feature)

        self.model_type = 'Transformer'

        encoder_layers = TransformerEncoderLayer(n_feature, n_head, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)

        self.n_feature = n_feature

        self.linear = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(n_feature, n_feature),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(n_feature, n_out),
        )


        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.img_encoder.weight.data.uniform_(-initrange, initrange)
        
    def preprocess(self, middles, cues, decisions):
        middles = self.img_encoder(middles)

        cues = self.cue_encoder(cues)
        decisions = self.decision_encoder(decisions)

        src = torch.stack((middles, cues, decisions), axis=0)

        return src

    def forward(self, middles, cues, decisions, middles_mask):
        # src.shape(sequence_length, batch_size, feature_number)
        # feedback:  0: AI == Human, 1: AI != Human, 2: Unknown, 3: [MASK]
        # action: 0: AI, 1: Human, 3: [MASK]
        src = self.preprocess(middles, cues, decisions)

        src = torch.sum(src, axis=0)

        src = src + self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=middles_mask)
        output = self.linear(output)
        return output.squeeze(2)
    
class SimpleTransformerEncoder_Access_Ablation(SimpleTransformerEncoder_Access):
    def __init__(self, n_feature=64, n_head=2, n_layers=3, n_hidden=1024, dropout=0.5, n_out=2, targets=['middles', 'cues', 'decisions']):
        super().__init__(n_feature, n_head, n_layers, n_hidden, dropout, n_out)
        self.targets=targets

        
    def preprocess(self, middles, cues, decisions):
        vec = list()
        if 'middles' in self.targets:
            vec.append(self.img_encoder(middles))
        if 'cues' in self.targets:
            vec.append(self.cue_encoder(cues))
        if 'decisions' in self.targets:
            vec.append(self.decision_encoder(decisions))

        src = torch.stack(vec, axis=0)

        return src
