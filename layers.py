"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2

# -----------------------------------------------------------------------------------------------------------------------

class WordCharEmbedding(nn.Module):
    """Embedding layer with both word and character-level component.
       Uses Gated Recurrent Unit (GRU) to generate character-level embeddings.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained character vectors.
        num_layers (int): Number of layers for GRU.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations

    Returns:
        emb (torch.Tensor): representation of all words in question or passage (depending on inputs)
    """
    def __init__(self, word_vectors, char_vectors, num_layers, cnn_size, hidden_size, drop_prob):
        super(WordCharEmbedding, self).__init__()

        self.hidden_size = hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding.from_pretrained(char_vectors)
        self.CNN = nn.Sequential(
            nn.Conv1d(in_channels=char_vectors.size(1),
                      out_channels=cnn_size,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(cnn_size)
        )
        self.proj = nn.Linear(word_vectors.size(1) + cnn_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)
        self.rnn = RNNEncoder(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

    def forward(self, w, c, mask):
        word_emb = self.word_embed(w)
        char_emb = self.char_embed(c)
        char_emb = char_emb.view(char_emb.shape[0] * char_emb.shape[1], char_emb.shape[3], char_emb.shape[2])
        char_emb = self.CNN(char_emb)
        char_emb = char_emb.view(word_emb.size(0), word_emb.size(1), char_emb.size(1)) # (batch_size, seq_length, hidden_size)
        emb = torch.cat((word_emb, char_emb), dim=2)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        result = self.rnn(emb, mask.sum(-1))
        return result


class GatedElementBasedRNNLayer(nn.Module):
    """Gated Element-Based RNN Layer.

    Args:
        passage_repr: passage representation
        question_repr: question representation

    Returns:
        vtp (torch.Tensor): setence-pair representation generated via soft-alignment of words
                            in the question and passage
    """
    def __init__(self, input_size, device, hidden_size, num_layers, drop_prob):
        super(GatedElementBasedRNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.device = device

        self.vT = nn.Linear(hidden_size, 1, bias=False)
        self.WuQ = nn.Linear(input_size, hidden_size, bias=False)
        self.WuP = nn.Linear(input_size, hidden_size, bias=False)
        self.WvP = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(2 * input_size, 2 * input_size, bias=False),
            nn.Sigmoid()
        )
        self.match_LSTM = nn.GRU(input_size=input_size * 2,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 num_layers=3,
                                 dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, passage_repr, question_repr, passage_mask, question_mask):
        self.match_LSTM.flatten_parameters()

        # Calculate ct
        question = self.WuQ(question_repr)
        passage = self.WuP(passage_repr)

        question_size, batch_size, _ = question_repr.size()
        passage_size = passage_repr.size(0)

        question = question.repeat(passage_size, 1, 1, 1).permute([1, 0, 2, 3])
        passage = passage.repeat(question_size, 1, 1, 1)
        question_mask = question_mask.view(question_size, 1, batch_size, 1)

        sj = self.vT(torch.tanh(question + passage))
        ai = masked_softmax(sj, question_mask, dim=0)
        expanded_q = question_repr.repeat(passage_size, 1, 1, 1).permute([1, 0, 2, 3])
        ct = (expanded_q * ai).sum(0)

        # Concatenate utp (passage_repr) and ct (attention-pooling vector)
        ct = torch.cat((passage_repr, ct), dim=2)

        # Apply Gate.
        ct = torch.mul(ct, self.gate(ct))

        result, _ = self.match_LSTM(ct)

        # Dropout
        result = F.dropout(result, self.drop_prob, self.training)
        return result # (num_words, batch_size, hidden_size)


class GatedElementBasedRNNLayer_Loop(nn.Module):
    """Gated Element-Based RNN Layer: Iterative approach.

    Returns:
        vtp (torch.Tensor): setence-pair representation generated via soft-alignment of words
                            in the question and passage
    """
    def __init__(self, input_size, device, hidden_size, num_layers, drop_prob):
        super(GatedElementBasedRNNLayer_Loop, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.device = device

        self.vT = nn.Linear(hidden_size, 1, bias=False)
        self.WuQ = nn.Linear(input_size, hidden_size, bias=False)
        self.WuP = nn.Linear(input_size, hidden_size, bias=False)
        self.WvP = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False),
            nn.Sigmoid()
        )

        self.cell = nn.GRUCell(input_size=2 * hidden_size,
                               hidden_size=hidden_size,
                               bias=False)

        self.match_LSTM = nn.GRU(input_size=hidden_size * 2,
                                 hidden_size=hidden_size,
                                 num_layers=3,
                                 dropout=drop_prob)

    def forward(self, passage_repr, question_repr, passage_mask, question_mask):
        _, batch_size, _ = question_repr.size()
        passage_size = passage_repr.size(0)

        prev = torch.zeros((batch_size, self.hidden_size)).to(self.device)
        result = torch.zeros((passage_size, batch_size, self.hidden_size)).to(self.device)
        question = self.WuQ(question_repr)
        passage = self.WuP(passage_repr)

        for i in range(passage_size):
            last_hidden_state = self.WvP(prev).unsqueeze(0)
            curr_passage = passage[i,:,:].to(self.device)
            sj = curr_passage.unsqueeze(0) + question + last_hidden_state
            sj = torch.tanh(sj)
            sj = self.vT(sj)
            ai = F.softmax(sj, dim=0)
            ct = (question * ai).sum(0)
            utct = torch.cat((curr_passage, ct), dim=-1)
            utct = utct * self.gate(utct)
            vt = self.cell(utct, prev)
            result[i,:,:] = vt
            prev = vt.to(self.device)

        return result

        
class SelfMatchingAttention(nn.Module):
    """Self-Matching Attention Layer. Directly matches question-aware passage representation against itself.

    Args:
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(SelfMatchingAttention, self).__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        self.vT = nn.Linear(hidden_size, 1, bias=False)
        self.WvP = nn.Linear(input_size, hidden_size, bias=False)
        self.WvPbar = nn.Linear(input_size, hidden_size, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(2 * input_size, 2 * input_size, bias=False),
            nn.Sigmoid()
        )

        self.AttentionRNN = nn.GRU(input_size=input_size * 2,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   num_layers=num_layers,
                                   dropout=drop_prob)

    def forward(self, passage):
        self.AttentionRNN.flatten_parameters()

        passage_size = passage.size(0)

        WvP = self.WvP(passage)
        WvPbar = self.WvPbar(passage)

        WvP = WvP.repeat(passage_size, 1, 1, 1).permute([1, 0, 2, 3])
        WvPbar = WvPbar.repeat(passage_size, 1, 1, 1)

        sj = self.vT(torch.tanh(WvP + WvPbar))
        ai = F.softmax(sj, dim=0)
        expanded_p = passage.repeat(passage_size, 1, 1, 1).permute([1, 0, 2, 3])
        ct = (expanded_p * ai).sum(0)

        # Concatenate utp (passage_repr) and ct (attention-pooling vector)
        ct = torch.cat((passage, ct), dim=2)

        # Apply Gate.
        ct = torch.mul(ct, self.gate(ct))

        result, _ = self.AttentionRNN(ct)

        # Dropout
        result = F.dropout(result, self.drop_prob, self.training)
        return result # (batch_size, num_words, hidden_size)



class SelfMatchingAttention_Loop(nn.Module):
    """Self-Matching Attention Layer. Directly matches question-aware passage representation against itself.

    Args:
    """
    def __init__(self, input_size, hidden_size, device, num_layers, drop_prob):
        super(SelfMatchingAttention_Loop, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

        self.vT = nn.Linear(hidden_size, 1, bias=False)
        self.WvP = nn.Linear(input_size, hidden_size, bias=False)
        self.WvPbar = nn.Linear(input_size, hidden_size, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False),
            nn.Sigmoid()
        )

        self.AttentionRNN = nn.GRU(input_size=hidden_size * 2,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   num_layers=3,
                                   dropout=drop_prob)

    def forward(self, passage_repr, passage_mask):
        passage_size, batch_size, _ = passage_repr.size()

        prev = torch.zeros((6, batch_size, self.hidden_size)).to(self.device)
        result = torch.zeros((passage_size, batch_size, 2 * self.hidden_size)).to(self.device)
        passage_repeat = self.WvP(passage_repr)
        passage = self.WvPbar(passage_repr)

        for i in range(passage_size):
            curr_passage = passage[i,:,:].to(self.device)
            sj = curr_passage.unsqueeze(0) + passage_repeat
            sj = torch.tanh(sj)
            sj = self.vT(sj)
            ai = F.softmax(sj, dim=0)
            ct = (passage * ai).sum(0)
            utct = torch.cat((curr_passage, ct), dim=-1)
            utct = utct * self.gate(utct)
            utct = utct.unsqueeze(0)
            vt, prev = self.AttentionRNN(utct, prev)
            result[i,:,:] = vt.squeeze(0)
            prev = prev.to(self.device)

        result = F.dropout(result, self.drop_prob, self.training)
        result = result.transpose(1, 0) # for bidaf
        return result


class RNetOutput(nn.Module):
    """Output layer used by R-Net for question answering. Uses pointer networks.

    Args:
        x (torch.Tensor): passage representation
    """
    def __init__(self, input_size, hidden_size, num_layers, drop_prob):
        super(RNetOutput, self).__init__()

        self.RNN = nn.GRUCell(input_size, input_size, False)

        self.WhA = nn.Linear(input_size, hidden_size, bias=False)
        self.WhP = nn.Linear(input_size, hidden_size, bias=False)
        self.vT = nn.Linear(hidden_size, 1, bias=False)
        
        self.question_transform = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True), # Vq included here.
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, h, q, passage_mask, question_mask):
        passage_size, batch_size, _ = h.size()
        question_size, _, _ = q.size()

        passage_mask = passage_mask.view((passage_size, batch_size, 1))
        question_mask = question_mask.view((question_size, batch_size, 1))

        initial = masked_softmax(q * self.question_transform(q), question_mask).sum(0)
        unsqueezed = initial.unsqueeze(0)
        ptr1 = self.vT(self.WhP(h) + self.WhA(unsqueezed))

        start = masked_softmax(ptr1, passage_mask, log_softmax=True)
        ct = masked_softmax(ptr1, passage_mask)

        ct = (ct * h).sum(0)
        hta = self.RNN(ct, initial).unsqueeze(0)

        ptr2 = self.vT(self.WhP(h) + self.WhA(hta))
        end = masked_softmax(ptr2, passage_mask, log_softmax=True)

        return start.transpose(0, 1).squeeze(-1), end.transpose(0, 1).squeeze(-1)
