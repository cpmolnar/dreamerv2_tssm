import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras import mixed_precision as prec

import common


class EnsembleTSSM(common.Module):

  def __init__(
      self, ensemble=5, stoch=30, deter=200, hidden=200, discrete=False,
      act='elu', norm='none', std_act='softplus', min_std=0.1):
    super().__init__()
    self._ensemble = ensemble
    self._stoch = stoch
    self._deter = deter
    self._hidden = hidden
    self._discrete = discrete
    self._act = common.get_act(act)
    self._norm = norm
    self._std_act = std_act
    self._min_std = min_std
    self._cell = common.GRUCell(self._deter, norm=True)
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)

  def initial(self, batch_size, seq_len=1):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, self._stoch, self._discrete], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, self._stoch], dtype),
          std=tf.zeros([batch_size, self._stoch], dtype),
          stoch=tf.zeros([batch_size, self._stoch], dtype),
          deter=self._cell.get_initial_state(None, batch_size, dtype))
    return state

  @tf.function
  def observe(self, embed, action, is_first, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    post, prior = self.obs_step(action, embed, is_first)
    return post, prior

  @tf.function
  def imagine(self, actions, states):
    for i in range(states['stoch'].shape[1], actions.shape[1]):
      prior = self.img_step(actions[:,:i], states['stoch'])
      states = {k: tf.concat([v, prior[k][:,None,-1]], axis=1) for k, v in states.items()}
    return states

  def get_feat(self, state):
    stoch = self._cast(state['stoch']) # z_t\hat
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1) # z_t\hat, h_t

  def get_dist(self, state, ensemble=False):
    if ensemble:
      state = self._suff_stats_ensemble(state['deter'])
    if self._discrete:
      logit = state['logit']
      logit = tf.cast(logit, tf.float32)
      dist = tfd.Independent(common.OneHotDist(logit), 1)
    else:
      mean, std = state['mean'], state['std']
      mean = tf.cast(mean, tf.float32)
      std = tf.cast(std, tf.float32)
      dist = tfd.MultivariateNormalDiag(mean, std)
    return dist

  def get_post_stochs(self, embeds, sample=True):
    x = self.get('obs_out', tfkl.Dense, self._hidden)(embeds)
    x = self.get('obs_out_norm', common.NormLayer, self._norm)(x)
    x = self._act(x)

    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    if sample=='both':
      return dist.sample(), dist.mode(), stats
    else: 
      stochs = dist.sample() if sample else dist.mode() # z_t | x_t
    return stochs, stats

  def get_prior_stochs(self, deter, sample=True):
    stats = self._suff_stats_ensemble(deter)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stochs = dist.sample() if sample else dist.mode()
    return stochs, stats

  @tf.function
  def obs_step(self, actions, embeds, is_first, sample=True):
    # if is_first.any():
    actions = tf.nest.map_structure(
        lambda x: tf.einsum('bl,bl...->bl...', 1.0 - is_first.astype(x.dtype), x), (actions))
    stochs, stochs_mode, stats = self.get_post_stochs(embeds, sample='both') # posterior: (z_t | x_t)
    prior = self.img_step(actions, tf.concat((tf.zeros_like(stochs[:,None,0]), stochs[:,:-1]), 1), sample) # prior: (z_t\hat | h_t), (h_t | z_t, a_t)
    post = {'stoch': stochs, 'deter': prior['deter'], **stats} # (z_t | x_t), (h_t | z_t, a_t)
    return post, prior

  @tf.function
  def img_step(self, actions, stochs, sample=True):
    if self._discrete:
      shape = stochs.shape[:-2] + [self._stoch * self._discrete]
      stochs = tf.reshape(stochs, shape)

    x = tf.concat([stochs, actions], -1) # z_t, a_t
    x = self.get('img_in', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', common.NormLayer, self._norm)(x)
    x = self._act(x)

    transformer_encoder = self.get('img_transformer', common.AttentionEncoder, num_layers=1, d_model=self._hidden, num_heads=1, dff=512, dropout_rate=0.1)
    deters = transformer_encoder(x)
    stochs, stats = self.get_prior_stochs(deters, sample=sample)
    prior = {'stoch': stochs, 'deter': deters, **stats}

    return prior

  def _suff_stats_ensemble(self, inp):
    bs = list(inp.shape[:-1])
    inp = inp.reshape([-1, inp.shape[-1]])
    stats = []
    for k in range(self._ensemble):
      x = self.get(f'img_out_{k}', tfkl.Dense, self._hidden)(inp)
      x = self.get(f'img_out_norm_{k}', common.NormLayer, self._norm)(x)
      x = self._act(x)
      stats.append(self._suff_stats_layer(f'img_dist_{k}', x))
    stats = {
        k: tf.stack([x[k] for x in stats], 0)
        for k, v in stats[0].items()}
    stats = {
        k: v.reshape([v.shape[0]] + bs + list(v.shape[2:]))
        for k, v in stats.items()}
    return stats

  def _suff_stats_layer(self, name, x):
    if self._discrete:
      x = self.get(name, tfkl.Dense, self._stoch * self._discrete, None)(x)
      logit = tf.reshape(x, x.shape[:-1] + [self._stoch, self._discrete])
      return {'logit': logit}
    else:
      x = self.get(name, tfkl.Dense, 2 * self._stoch, None)(x)
      mean, std = tf.split(x, 2, -1)
      std = {
          'softplus': lambda: tf.nn.softplus(std),
          'sigmoid': lambda: tf.nn.sigmoid(std),
          'sigmoid2': lambda: 2 * tf.nn.sigmoid(std / 2),
      }[self._std_act]()
      std = std + self._min_std
      return {'mean': mean, 'std': std}

  def kl_loss(self, post, prior, forward, balance, free, free_avg):
    kld = tfd.kl_divergence
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    lhs, rhs = (prior, post) if forward else (post, prior)
    mix = balance if forward else (1 - balance)
    if balance == 0.5:
      value = kld(self.get_dist(lhs), self.get_dist(rhs))
      loss = tf.maximum(value, free).mean()
    else:
      value_lhs = value = kld(self.get_dist(lhs), self.get_dist(sg(rhs)))
      value_rhs = kld(self.get_dist(sg(lhs)), self.get_dist(rhs))
      if free_avg:
        loss_lhs = tf.maximum(value_lhs.mean(), free)
        loss_rhs = tf.maximum(value_rhs.mean(), free)
      else:
        loss_lhs = tf.maximum(value_lhs, free).mean()
        loss_rhs = tf.maximum(value_rhs, free).mean()
      loss = mix * loss_lhs + (1 - mix) * loss_rhs
    return loss, value

class PolicyMLP(common.Module):

  def __init__(self, shape, layers, units, act='elu', norm='none', **out):
    self._shape = (shape,) if isinstance(shape, int) else shape
    self._layers = layers
    self._units = units
    self._norm = norm
    self._act = common.nets.get_act(act)
    self._out = out

  def __call__(self, features, em=None):
    reshape = len(features.shape)==2
    if reshape: features=features[None,:]
    mem_seq = em.retrieve_seqs(features)[0]
    features = tf.concat((tf.stop_gradient(features), mem_seq), -1)

    x = tf.cast(features, prec.global_policy().compute_dtype)
    x = x.reshape([-1, x.shape[-1]])
    for index in range(self._layers):
      x = self.get(f'dense{index}', tfkl.Dense, self._units)(x)
      x = self.get(f'norm{index}', common.nets.NormLayer, self._norm)(x)
      x = self._act(x)
    x = x.reshape(features.shape[:-1] + [x.shape[-1]])
    x = x[0] if reshape else x
    return self.get('out', common.nets.DistLayer, self._shape, **self._out)(x)