"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import pdb


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, device, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.device = device
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.wordcharembed = layers.WordCharEmbedding(word_vectors=word_vectors,
                                            char_vectors=char_vectors,
                                            cnn_size=16,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.selfatt = layers.SelfMatchingAttention_Loop(input_size=8 * hidden_size,
                                                hidden_size=hidden_size,
                                                num_layers=1,
                                                device=self.device,
                                                drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_enc = self.wordcharembed(cw_idxs, cc_idxs)         # (batch_size, c_len, 2 * hidden_size)

        q_enc = self.wordcharembed(qw_idxs, qc_idxs)         # (batch_size, q_len, 2 * hidden_size)


        # c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        # q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        # c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        # q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        h_p = self.selfatt(att, c_mask)

        h_p = h_p.permute([1, 0, 2])

        mod = self.mod(h_p, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(h_p, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        torch.cuda.empty_cache()

        return out


class RNet(nn.Module):
    """R-Net Model for SQuAD.

    Based on the paper:
    "R-Net: Machine Reading Comprehension with Self-Matching Networks"
    by Natural Language Computing Group, Microsoft Research Asia
    (https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf).

    Uses the four layers within the model:
        - Bidirectional Recurrent Network Layer: Processes question and passage separately.
        - Gated Attention-Based Recurrent Network: Match question and passage, obtaining question-aware representation for passage.
        - Self-Matching Attention Layer: Aggregate evidence from the whole passage and refine the passage representation.
        - Output layer: Pointer networks + attention-pooling over question representation.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, batch_size, device, hidden_size, drop_prob=0.):
        super(RNet, self).__init__()
        self.num_layers = 1

        self.emb2 = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)
                                     
        self.emb = layers.WordCharEmbedding(word_vectors=word_vectors,
                                            char_vectors=char_vectors,
                                            cnn_size=16,
                                            hidden_size=hidden_size,
                                            num_layers=self.num_layers,
                                            drop_prob=drop_prob)

        self.gated_rnn = layers.GatedElementBasedRNNLayer(input_size=2 * hidden_size, # bidirectional output from prev layer
                                                          hidden_size=hidden_size,
                                                          device=device,
                                                          num_layers=self.num_layers,
                                                          drop_prob=drop_prob)

        self.bidafatt = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.att = layers.SelfMatchingAttention(input_size=8 * hidden_size,
                                                hidden_size=hidden_size,
                                                device=device,
                                                num_layers=self.num_layers,
                                                drop_prob=drop_prob)

        self.out = layers.RNetOutput(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=self.num_layers,
                                     drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=2 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.bidafout = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)


    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs, c_mask)        

        q_emb = self.emb(qw_idxs, qc_idxs, q_mask)    

        # cc = self.emb2(cw_idxs)        # (batch_size, c_len, hidden_size)
        # qq = self.emb2(qw_idxs)       # (batch_size, q_len, hidden_size)

        # c_emb = self.enc(cc, c_len).transpose(0, 1)     # (batch_size, c_len, 2 * hidden_size)
        # q_emb = self.enc(qq, q_len).transpose(0, 1)     # (batch_size, q_len, 2 * hidden_size)

        v_p = self.gated_rnn(c_emb, q_emb, c_mask, q_mask)

        h_p = self.att(v_p, c_mask)

        start, end = self.out(h_p, q_emb.transpose(1, 0), c_mask, q_mask)

        # h_p = h_p.transpose(0, 1)

        # mod = self.mod(h_p, c_len)        # (batch_size, c_len, 2 * hidden_size)

        # start, end = self.bidafout(h_p, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        torch.cuda.empty_cache()

        return start, end
