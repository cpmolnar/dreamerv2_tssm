import random
import numpy as np
import tensorflow as tf
from tensorflow import nn
import tensorflow.keras.layers as tfkl

import pathlib, pickle
import common

class ScaledDotProductAttention(common.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, lamb):
        super().__init__()
        self.lamb = lamb

    def __call__(self, q, k, v, mask=None, method='default'):
        q = tf.transpose(q, [1, 0, 2])
        k = tf.transpose(k, [1, 2, 0])
        v = tf.transpose(v, [1, 0, 2])
        attn = tf.matmul(q, k)
        # attn = tfkl.Softmax(axis=-1)(attn)
        # output = tf.matmul(attn, v)

        gather = tf.gather(v[0], tf.argmax(attn, axis=-1)[0])
        output = tf.where(
            tf.broadcast_to(tf.expand_dims(attn[0].max(axis=-1), 1), gather.shape) > self.lamb, 
            gather, 
            tf.zeros_like(gather)
        )

        return output, attn
    
class RaggedMultiHeadAttention(common.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_k, d_v, lamb):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = tfkl.Dense(n_head * d_k, use_bias=False)
        self.w_ks = tfkl.Dense(n_head * d_k, use_bias=False)
        # self.w_vs = tfkl.Dense(n_head * d_v, use_bias=False)
        # self.fc = tfkl.Dense(d_v, use_bias=False)

        self.attention = ScaledDotProductAttention(lamb=lamb)

        # self.layer_norm = tfkl.LayerNormalization(epsilon=1e-6)


    def __call__(self, q, k, v, mask=None, method='default'):
        batch_size, seq_len, n_feats = q.shape
        q = q.reshape([batch_size * seq_len] + [n_feats])
        k = k.merge_dims(0, 1).to_tensor()
        v = v.merge_dims(0, 1).to_tensor()
        
        # residual = q


        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        q = self.w_qs(q).reshape((-1, n_head, d_k))
        k = self.w_ks(k).reshape((-1, n_head, d_k))
        v = v.reshape((-1, n_head, d_v))
        q, attn = self.attention(q, k, v, mask=mask, method=method)   

        # q = self.fc(q)
        # q += residual
        # q = self.layer_norm(q) 

        return q.reshape([batch_size, seq_len, n_feats]), attn

class RaggedEpisodicMemory(common.Module):
    def __init__(self, seq_len, d_k, d_v, max_size, verbose=False):
        super().__init__()
        self.max_size = max_size
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.verbose = verbose

        self.k = tf.ragged.constant([], ragged_rank=1, inner_shape=(self.d_k,), dtype=tf.float32, name='keys')
        self.v = tf.ragged.constant([], ragged_rank=1, inner_shape=(self.d_v,), dtype=tf.float32, name='values')

        self.cleanup_ctr = 0
        self.cleanup_every = 10

        self.drop_window = 0

        self.lamb = 0.9

    def __len__(self):
        return self.k.shape[0]

    def add_seqs(self, k, v, n=2):
        if len(self)==self.max_size: return

        v = tf.squeeze(v)
        idxs = tf.where(v[:,1:].sum(axis=1)>0)
        n_seqs = len(idxs)
        if n_seqs<n: return
        idxs = tf.squeeze(idxs)
        v = tf.gather(v, idxs)
        k = tf.gather(k, idxs)
        
        
        target_idxs = tf.argmax(v[:,1:], axis=-1) + 1
        targets = tf.gather_nd(k, tf.stack((tf.range(n_seqs, dtype=tf.int64), target_idxs), 1))

        values = tf.map_fn(
            lambda i: tf.RaggedTensor.from_tensor(tf.repeat(tf.expand_dims(targets[i], 0), target_idxs[i], axis=0)), 
            tf.range(n_seqs), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(None, self.d_v), ragged_rank=1), 
            dtype=tf.float32
        )

        keys = tf.map_fn(
            lambda i: tf.RaggedTensor.from_tensor(k[i][:target_idxs[i]]), 
            tf.range(n_seqs), 
            fn_output_signature=tf.RaggedTensorSpec(shape=(None, self.d_k), ragged_rank=1), 
            dtype=tf.float32
        )

        value_rewards = tf.map_fn(lambda i: v[i][:target_idxs[i]+1].sum(), tf.range(n_seqs), fn_output_signature=tf.float32)
        _, idxs_to_add = tf.math.top_k(value_rewards, k=n)

        keys   = tf.map_fn(lambda i: keys[i]  , idxs_to_add, fn_output_signature=tf.RaggedTensorSpec(shape=(None, self.d_k), ragged_rank=1))
        self.k = tf.concat((self.k, keys), 0)

        values = tf.map_fn(lambda i: values[i], idxs_to_add, fn_output_signature=tf.RaggedTensorSpec(shape=(None, self.d_v), ragged_rank=1))
        self.v = tf.concat((self.v, values), 0)

        if self.verbose and n > 0: print(f'Added {n} sequences to memory. Memory table has {len(self)}/{self.max_size} sequences left.')

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

    def retrieve_skill(self, q, stopping_criteria=None):
        if len(self)==0: return tf.zeros_like(q)
        mem_self_attn = self.get('mem_self_attn', RaggedMultiHeadAttention, n_head=1, d_k=self.d_k, d_v=self.d_v, lamb=self.lamb)
        mem_output, mem_attn = mem_self_attn(q, self.k, self.v, method='full_table')
        return mem_output#, mem_attn

    def save(self, filepath):
        print(f'Save em checkpoint with {len(self)} episodes.')
        with pathlib.Path(filepath / 'em_keys.pkl')  .open('wb') as f: pickle.dump(self.k, f)
        with pathlib.Path(filepath / 'em_values.pkl').open('wb') as f: pickle.dump(self.v, f)

    def load(self, filepath):
        with pathlib.Path(filepath / 'em_keys.pkl')  .open('rb') as f: self.k = pickle.load(f)
        with pathlib.Path(filepath / 'em_values.pkl').open('rb') as f: self.v = pickle.load(f)
        print(f'Load checkpoint with {len(self)} episodes.')