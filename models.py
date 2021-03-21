"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import pdb

class BiDAF_RNet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_RNet, self).__init__()
        self.emb = layers.WordCharEmbedding(word_vectors=word_vectors,
                                            char_vectors=char_vectors,
                                            cnn_size=16,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.selfatt = layers.SelfMatchingAttention(input_size=8 * hidden_size,
                                                hidden_size=4 * hidden_size,
                                                num_layers=3,
                                                drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        
        c_enc = self.emb(cw_idxs, cc_idxs, c_mask)        
        q_enc = self.emb(qw_idxs, qc_idxs, q_mask)

        att = self.att(c_enc, q_enc, c_mask, q_mask)

        h_p = self.selfatt(att)

        mod = self.mod(h_p, c_mask.sum(-1))

        out = self.out(att, mod, c_mask)

        torch.cuda.empty_cache()

        return out
