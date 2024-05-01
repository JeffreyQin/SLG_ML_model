from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_positional=True, relative_positional_distance=100):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
       
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True, relative_positional_distance=100):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    d_qkv = d_model // n_head
    assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
    self.d_qkv = d_qkv

    self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
    nn.init.xavier_normal_(self.w_q)
    nn.init.xavier_normal_(self.w_k)
    nn.init.xavier_normal_(self.w_v)
    nn.init.xavier_normal_(self.w_o)

    self.dropout = nn.Dropout(dropout)

    if relative_positional:
        self.relative_positional = LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, d_qkv, True)
    else:
        self.relative_positional = None

  def forward(self, x):
    """Runs the multi-head self-attention layer.

    Args:
      x: the input to the layer, a tensor of shape [length, batch_size, d_model]
    Returns:
      A single tensor containing the output from this layer
    """

    q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
    k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
    v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
    logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

    if self.relative_positional is not None:
        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        logits = logits + position_logits.view(b,h,l,l)

    probs = F.softmax(logits, dim=-1)
    probs = self.dropout(probs)
    o = torch.einsum('bhqk,bhka->bhqa', probs, v)
    out = torch.einsum('bhta,haf->tbf', o, self.w_o)
    return out

class LearnedRelativePositionalEmbedding(nn.Module):

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
       
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
      
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
    
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        # mask out tokens out of range
        length = query.size(0)
        if length > self.max_relative_pos:
            # there is some padding
            pad_length = length - self.max_relative_pos
            positional_logits[:,:,:pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:,:,-pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
      
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]
