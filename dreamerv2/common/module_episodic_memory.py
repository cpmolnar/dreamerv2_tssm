import random
import numpy as np
import tensorflow as tf
from tensorflow import nn
import tensorflow.keras.layers as tfkl

import pathlib, pickle
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
            q = tf.transpose(q, [1, 0, 2])
            k = tf.transpose(k, [1, 2, 0])
            v = tf.transpose(v, [1, 0, 2])
            attn = tf.matmul(q / self.temperature, k)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(tfkl.Softmax(axis=-1)(attn))
            if combination=='default': output = tf.matmul(attn, v)
            elif combination=='absolute': 
                output = tf.gather(v[0], tf.argmax(attn, axis=-1)[0])
        else:
            attn = tf.matmul(q / self.temperature, k.transpose(2, 3)) # batch_size, n_heads, query_seq_len, key_seq_len
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(tfkl.Softmax(attn, dim=-1))
            output = tf.matmul(attn, v)

        return output, attn
    
class RaggedMultiHeadAttention(common.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # self.w_qs = tfkl.Dense(n_head * d_k, use_bias=False)
        # self.w_ks = tfkl.Dense(n_head * d_k, use_bias=False)
        # self.w_vs = tfkl.Dense(n_head * d_v, use_bias=False)
        # self.fc = tfkl.Dense(d_v, use_bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = tfkl.Dropout(dropout)
        # self.layer_norm = tfkl.LayerNormalization(epsilon=1e-6)


    def __call__(self, q, k, v, mask=None, method='default', combination='default'):
        batch_size, seq_len, n_feats = q.shape
        q = q.reshape([batch_size * seq_len] + [n_feats])
        k = tf.concat([i.to_tensor() for i in k], 0)
        v = tf.concat([i.to_tensor() for i in v], 0)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, sz_t,  = q.shape[0], q.shape[1], k.shape[0]

        # residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # q = self.w_qs(q).reshape((sz_b, n_head, d_k))
        # k = self.w_ks(k).reshape((sz_t, n_head, d_k))
        # v = self.w_vs(v).reshape((sz_t, n_head, d_v))
        q = q.reshape((sz_b, n_head, d_k))
        k = k.reshape((sz_t, n_head, d_k))
        v = v.reshape((sz_t, n_head, d_v))

        # Transpose for attention dot product: b x n x dv
        # q, k, v = tf.transpose(q, [0, 2, 1, 3]), tf.transpose(k, [0, 2, 3]), tf.transpose(v, [0, 2, 3])

        if mask is not None:
            mask = tf.expand_dims(mask, 1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask, method=method, combination=combination)    

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        # q = q.reshape((sz_b, len_q, -1))
        # q = self.dropout(self.fc(q))
        # q += residual

        # q = self.layer_norm(q)

        return q.reshape([batch_size, seq_len, n_feats]), attn
    

# class MultiHeadAttention(common.Module):
#     ''' Multi-Head Attention module '''

#     def __init__(self, n_head, d_k, d_v, dropout=0.1):
#         super().__init__()

#         self.n_head = n_head
#         self.d_k = d_k
#         self.d_v = d_v

#         self.w_qs = tfkl.Dense(n_head * d_k, use_bias=False)
#         self.w_ks = tfkl.Dense(n_head * d_k, use_bias=False)
#         self.w_vs = tfkl.Dense(n_head * d_v, use_bias=False)
#         self.fc = tfkl.Dense(d_v, use_bias=False)

#         self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

#         self.dropout = tfkl.Dropout(dropout)
#         self.layer_norm = tfkl.LayerNormalization(epsilon=1e-6)


#     def __call__(self, q, k, v, mask=None, method='default', combination='default'):
#         d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
#         sz_b, len_q, sz_t, len_k, len_v = q.shape[0], q.shape[1], k.shape[0], k.shape[1], v.shape[1]

#         # residual = q

#         # Pass through the pre-attention projection: b x lq x (n*dv)
#         # Separate different heads: b x lq x n x dv
#         q = self.w_qs(q).reshape((sz_b, len_q, n_head, d_k))
#         k = self.w_ks(k).reshape((sz_t, len_k, n_head, d_k))
#         v = self.w_vs(v).reshape((sz_t, len_v, n_head, d_v))

#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = tf.transpose(q, [0, 2, 1, 3]), tf.transpose(k, [0, 2, 1, 3]), tf.transpose(v, [0, 2, 1, 3])

#         if mask is not None:
#             mask = tf.expand_dims(mask, 1)   # For head axis broadcasting.
#         q, attn = self.attention(q, k, v, mask=mask, method=method, combination=combination)    

#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = tf.transpose(q, [0, 2, 1, 3]).reshape((sz_b, len_q, -1))
#         q = self.dropout(self.fc(q))
#         # q += residual

#         q = self.layer_norm(q)

#         return q, attn

class RaggedEpisodicMemory(common.Module):
    def __init__(self, seq_len, d_k, d_v, max_size, verbose=False, dropout=0.1):
        super().__init__()
        self.max_size = max_size
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.verbose = verbose

        self.k = tf.ragged.constant([], ragged_rank=2, dtype=tf.float32)
        self.v = tf.ragged.constant([], ragged_rank=2, dtype=tf.float32)
        # self.f = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)
        # self.i = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)

        self.cleanup_ctr = 0
        self.cleanup_every = 10

        self.drop_window = 0

    def __len__(self):
        return self.k.shape[0]

    def add_seqs(self, k, v, n=2):
        if len(self)==self.max_size: return
        batch_size, seq_len, _ = k.shape

        v = tf.squeeze(v)
        idxs = tf.where(v[:,1:].sum(axis=1)>0)
        if len(idxs)<n: return
        else: idxs = tf.squeeze(idxs)
        v = tf.gather(v, idxs)
        k = tf.gather(k, idxs)
        
        
        target_idxs = tf.argmax(v[:,1:], axis=-1) + 1
        targets = tf.gather_nd(k, tf.stack((tf.range(len(k), dtype=tf.int64), target_idxs), 1))

        values = tf.ragged.stack([tf.repeat(targets[i][None,:], target_idx, 0) for i, target_idx in enumerate(target_idxs)])
        keys = tf.ragged.stack([k[i][:target_idxs[i]] for i in np.arange(len(idxs))])

        value_rewards = tf.concat([v[i, :target_idx+1].sum()[None,] for i, target_idx in enumerate(target_idxs)], 0)
        _, idxs_to_add = tf.math.top_k(value_rewards, k=n)

        for idx in range(n):
            self.k = tf.concat((self.k, keys  [None, idxs_to_add[idx]]), 0)
            self.v = tf.concat((self.v, values[None, idxs_to_add[idx]]), 0)

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
        self.k = self.k[n:]
        self.v = self.v[n:]
        n_after = len(self)
            
        if self.verbose and n_before - n_after > 0: 
            print(f'Removed {n_before - n_after} sequences from memory. Memory table has {n_after}/{self.max_size} sequences left.')

    def retrieve_skill(self, q):
        if len(self)==0: return tf.zeros_like(q)
        mem_self_attn = self.get('mem_self_attn', RaggedMultiHeadAttention, n_head=1, d_k=self.d_k, d_v=self.d_v, dropout=0.1)
        mem_output, mem_attn = mem_self_attn(q, self.k, self.v, method='full_table', combination='absolute')
        return mem_output#, mem_attn
    

# class EpisodicMemory(common.Module):
#     def __init__(self, seq_len, d_k, d_v, max_size, verbose=False, dropout=0.1):
#         super().__init__()
#         self.max_size = max_size
#         self.seq_len = seq_len
#         self.d_k = d_k
#         self.d_v = d_v
#         self.verbose = verbose

#         self.k = tf.Variable(tf.zeros(shape=(max_size, seq_len, d_k)), trainable=False)
#         self.v = tf.Variable(tf.zeros(shape=(max_size, seq_len, d_v)), trainable=False)
#         self.f = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)
#         self.i = tf.Variable(tf.zeros(shape=(max_size, seq_len)), trainable=False)
#         self.mask = tf.Variable(tf.zeros(shape=(max_size,), dtype=np.bool), trainable=False)

#         self.cleanup_ctr = 0
#         self.cleanup_every = 10

#         self.drop_window = 0

#     def __len__(self):
#         return np.count_nonzero(self.mask)

#     def add_seqs(self, k, v, n=2):
#         if len(self)==self.max_size: return
#         batch_size, seq_len, _ = k.shape
#         # if n == None: n = batch_size
#         # if len(self) + n > self.max_size: n = self.max_size - len(self)
#         # idxs_to_add = np.random.permutation(np.arange(batch_size))[:n]
#         # add_pos = np.arange(self.max_size)[~self.mask][:n]

#         # values, _ = tf.raw_ops.UniqueV2(x=v[:,:,-1], axis=[0])
#         idxs = [any(i>0) for i in v]
#         idxs = np.arange(batch_size)[idxs]
#         target_idxs = [tf.argmax(i[:,0]).numpy() for i in tf.gather(v, idxs)]
#         targets = [k[i,target_idx].numpy() for i, target_idx in enumerate(target_idxs)]

#         values = [tf.repeat(targets[i][None,:], target_idx-1, 1) for i, target_idx in enumerate(target_idxs)]
#         keys = [k[i][:target_idxs[i]-1] for i in np.arange(len(idxs))]

#         value_rewards = tf.concat([tf.gather(v, idxs)[i, :target_idx+1].sum()[None,] for i, target_idx in enumerate(target_idxs)], 0)
#         _, idxs_to_add = tf.math.top_k(value_rewards, k=n)
#         _, idxs_to_replace = tf.math.top_k(self.v[:,:,-1].sum(axis=1), k=n)

#         for idx in range(n):
#             self.k[idxs_to_replace[idx]].assign(k[idxs_to_add[idx]])
#             self.v[idxs_to_replace[idx]].assign(v[idxs_to_add[idx]])
#             self.mask[idxs_to_replace[idx]].assign(True)

#         if self.verbose and n > 0: print(f'Added {n} sequences to memory. Memory table has {len(self)}/{self.max_size} sequences left.')

#     # def update_freq(self, f, i):
#     #     seq_len = i.shape[1]
#     #     f = f.sum(dim=1)

#     #     with torch.no_grad():
#     #         f_add = torch.zeros_like(self.f)
#     #         f_add[:, :seq_len] += f.sum(dim=0).T

#     #         i_add = torch.zeros_like(self.i)
#     #         i_add[:, :seq_len] += (f[:, :seq_len] * i[...,None].repeat(1, 1, self.max_size)).sum(dim=0).T

#     #         self.f = self.f + f_add
#     #         self.i = self.i + i_add

#     def cleanup(self, n=8):
#         if (len(self) != self.max_size): return
#         self.cleanup_ctr += 1
#         if self.cleanup_ctr % self.cleanup_every != 0: return

#         n_before = len(self)

#         # avg_loss_gain = torch.where(self.f!=0, self.i/self.f, self.f)
#         # seqs_to_drop = torch.topk(avg_loss_gain.sum(dim=-1), largest=False, k=botk)[1]

#         _, idxs_to_drop = tf.math.top_k(-tf.abs(self.v[:,:,-1]).sum(axis=1), k=n)
#         for idx in idxs_to_drop:
#             self.k[idx].assign(tf.zeros(shape=(self.seq_len, self.d_k)))
#             self.v[idx].assign(tf.zeros(shape=(self.seq_len, self.d_v)))
#             self.mask[idx].assign(False)
            
#         if self.verbose and n_before - len(self) > 0: 
#             print(f'Removed {n_before - len(self)} sequences from memory. Memory table has {len(self)}/{self.max_size} sequences left.')
#             print(f'The idx of sequences removed are {str(idxs_to_drop)}')

#     def retrieve_skill(self, q):
#         mem_self_attn = self.get('mem_self_attn', MultiHeadAttention, n_head=1, d_k=self.d_k, d_v=self.d_v, dropout=0.1)
#         mem_output, mem_attn = mem_self_attn(q, self.k, self.v, method='full_table', combination='absolute')
#         return mem_output#, mem_attn

    def save(self, filepath):
        print(f'Save em checkpoint with {len(self)} episodes.')
        with pathlib.Path(filepath / 'em_keys.pkl')  .open('wb') as f: pickle.dump(self.k, f)
        with pathlib.Path(filepath / 'em_values.pkl').open('wb') as f: pickle.dump(self.v, f)

    def load(self, filepath):
        with pathlib.Path(filepath / 'em_keys.pkl')  .open('rb') as f: self.k = pickle.load(f)
        with pathlib.Path(filepath / 'em_values.pkl').open('rb') as f: self.v = pickle.load(f)
        print(f'Load checkpoint with {len(self)} episodes.')