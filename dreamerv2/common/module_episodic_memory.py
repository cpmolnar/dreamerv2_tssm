import random
import numpy as np
import tensorflow as tf
from tensorflow import nn
import tensorflow.keras.layers as tfkl

import common

class ScaledDotProductAttention(common.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = tfkl.Dropout(attn_dropout)

    def __call__(self, q, k, v, mask=None, method='default', combination='default'):
        if method=='broadcast':
            q, k, v = tf.transpose(q, [1, 2, 0, 3]), tf.transpose(k, [1, 2, 3, 0]), tf.transpose(v, [1, 2, 0, 3])
            attn = tf.matmul(q / self.temperature, k)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(tfkl.Softmax(axis=-1)(attn))
            output = tf.matmul(attn, v)
            output, attn = tf.transpose(output, [2, 0, 1, 3]), tf.transpose(attn, [2, 0, 1, 3])
        elif method=='full_table':
            q_len, num_heads, q_seq_len, q_feats = q.shape
            k_len, _, k_seq_len, k_feats = k.shape
            v_len, _, _, v_feats = v.shape
            q = tf.transpose(q, [1, 2, 0, 3]).reshape((num_heads, q_seq_len * q_len, q_feats))
            k = tf.transpose(k, [1, 3, 2, 0]).reshape((num_heads, k_feats, k_seq_len * k_len))
            v = tf.transpose(v, [1, 2, 0, 3]).reshape((num_heads, k_seq_len * v_len, v_feats))
            attn = tf.matmul(q / self.temperature, k)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(tfkl.Softmax(axis=-1)(attn))
            if combination=='default': output = tf.matmul(attn, v)
            elif combination=='absolute': output = tf.gather(v[0], tf.argmax(attn, axis=-1)[0])
            output, attn =  tf.transpose(output.reshape((num_heads, q_seq_len, q_len, v_feats)), [2, 0, 1, 3]), \
                            tf.transpose(attn.reshape((num_heads, q_seq_len, q_len, k_seq_len * v_len)), [2, 0, 1, 3])
        else:
            attn = tf.matmul(q / self.temperature, k.transpose(2, 3)) # batch_size, n_heads, query_seq_len, key_seq_len
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(tfkl.Softmax(attn, dim=-1))
            output = tf.matmul(attn, v)

        return output, attn

class MultiHeadAttention(common.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = tfkl.Dense(n_head * d_k, use_bias=False)
        self.w_ks = tfkl.Dense(n_head * d_k, use_bias=False)
        self.w_vs = tfkl.Dense(n_head * d_v, use_bias=False)
        self.fc = tfkl.Dense(d_v, use_bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = tfkl.Dropout(dropout)
        self.layer_norm = tfkl.LayerNormalization(epsilon=1e-6)


    def __call__(self, q, k, v, mask=None, method='default', combination='default'):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, sz_t, len_k, len_v = q.shape[0], q.shape[1], k.shape[0], k.shape[1], v.shape[1]

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).reshape((sz_b, len_q, n_head, d_k))
        k = self.w_ks(k).reshape((sz_t, len_k, n_head, d_k))
        v = self.w_vs(v).reshape((sz_t, len_v, n_head, d_v))

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = tf.transpose(q, [0, 2, 1, 3]), tf.transpose(k, [0, 2, 1, 3]), tf.transpose(v, [0, 2, 1, 3])

        if mask is not None:
            mask = tf.expand_dims(mask, 1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask, method=method, combination=combination)    

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = tf.transpose(q, [0, 2, 1, 3]).reshape((sz_b, len_q, -1))
        q = self.dropout(self.fc(q))
        # q += residual

        q = self.layer_norm(q)

        return q, attn

class EpisodicMemory(common.Module):
    def __init__(self, seq_len, d_k, d_v, max_size, verbose=False, dropout=0.1):
        super().__init__()
        self.max_size = max_size
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.verbose = verbose

        self.k = tf.Variable(tf.zeros(shape=(max_size, seq_len, d_k)), trainable=False)
        self.v = tf.Variable(tf.zeros(shape=(max_size, seq_len, d_v)), trainable=False)
        self.f = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)
        self.i = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)
        self.mask = tf.Variable(tf.zeros(shape=(max_size,), dtype=np.bool), trainable=False)

        self.cleanup_ctr = 0
        self.cleanup_every = 10

        self.drop_window = 0

    def __len__(self):
        return np.count_nonzero(self.mask)

    def add_seqs(self, k, v, n=2):
        if len(self)==self.max_size: return
        batch_size, seq_len, _ = k.shape
        # if n == None: n = batch_size
        # if len(self) + n > self.max_size: n = self.max_size - len(self)
        # idxs_to_add = np.random.permutation(np.arange(batch_size))[:n]
        # add_pos = np.arange(self.max_size)[~self.mask][:n]

        values, _ = tf.raw_ops.UniqueV2(x=v[:,:,-1], axis=[0])
        _, idxs_to_add = tf.math.top_k(tf.abs(values).sum(axis=1), k=n)
        _, idxs_to_replace = tf.math.top_k(-tf.abs(self.v[:,:,-1]).sum(axis=1), k=n)

        for idx in range(n):
            self.k[idxs_to_replace[idx]].assign(k[idxs_to_add[idx]])
            self.v[idxs_to_replace[idx]].assign(v[idxs_to_add[idx]])
            self.mask[idxs_to_replace[idx]].assign(True)

        if self.verbose and n > 0: print(f'Added {n} sequences to memory. Memory table has {len(self)}/{self.max_size} sequences left.')

    # def update_freq(self, f, i):
    #     seq_len = i.shape[1]
    #     f = f.sum(dim=1)

    #     with torch.no_grad():
    #         f_add = torch.zeros_like(self.f)
    #         f_add[:, :seq_len] += f.sum(dim=0).T

    #         i_add = torch.zeros_like(self.i)
    #         i_add[:, :seq_len] += (f[:, :seq_len] * i[...,None].repeat(1, 1, self.max_size)).sum(dim=0).T

    #         self.f = self.f + f_add
    #         self.i = self.i + i_add

    def cleanup(self, n=8):
        if (len(self) != self.max_size): return
        self.cleanup_ctr += 1
        if self.cleanup_ctr % self.cleanup_every != 0: return

        n_before = len(self)

        # avg_loss_gain = torch.where(self.f!=0, self.i/self.f, self.f)
        # seqs_to_drop = torch.topk(avg_loss_gain.sum(dim=-1), largest=False, k=botk)[1]

        _, idxs_to_drop = tf.math.top_k(-tf.abs(self.v[:,:,-1]).sum(axis=1), k=n)
        for idx in idxs_to_drop:
            self.k[idx].assign(tf.zeros(shape=(self.seq_len, self.d_k)))
            self.v[idx].assign(tf.zeros(shape=(self.seq_len, self.d_v)))
            self.mask[idx].assign(False)
            
        if self.verbose and n_before - len(self) > 0: 
            print(f'Removed {n_before - len(self)} sequences from memory. Memory table has {len(self)}/{self.max_size} sequences left.')
            print(f'The idx of sequences removed are {str(idxs_to_drop)}')

    def retrieve_seqs(self, q):
        mem_self_attn = self.get('mem_self_attn', MultiHeadAttention, n_head=1, d_k=self.d_k, d_v=self.d_v, dropout=0.1)
        mem_output, mem_attn = mem_self_attn(q, self.k, self.v, method='full_table', combination='absolute')
        return mem_output, mem_attn