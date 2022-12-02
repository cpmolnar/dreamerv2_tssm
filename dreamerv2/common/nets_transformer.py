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
    self._cast = lambda x: tf.cast(x, prec.global_policy().compute_dtype)
    self.transformer_loss = None
    self.transformer_opt = common.Optimizer('transformer', **{'opt': 'adam', 'lr': 8e-5, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6})

  def initial(self, batch_size, seq_len=1):
    dtype = prec.global_policy().compute_dtype
    if self._discrete:
      state = dict(
          logit=tf.zeros([batch_size, seq_len, self._stoch, self._discrete], dtype),
          stoch=tf.zeros([batch_size, seq_len, self._stoch, self._discrete], dtype),
          deter=tf.zeros([batch_size, seq_len, self._deter], dtype))
    else:
      state = dict(
          mean=tf.zeros([batch_size, seq_len, self._stoch], dtype),
          std=tf.zeros([batch_size, seq_len, self._stoch], dtype),
          stoch=tf.zeros([batch_size, seq_len, self._stoch], dtype),
          deter=tf.zeros([batch_size, seq_len, self._deter], dtype))
    return state

  @tf.function
  def observe(self, embeds, actions, is_first):
    '''
    input:
      embeds:       x_t
      actions:      a_t
      state.stoch:  z_t\hat
      state.deter:  h_t\hat
      state.logit:
    output:
      post.stoch:   z_t
      post.deter:   h_t
      prior.stoch:  z_t\hat
      prior.deter:  h_t
    '''
    # if states is None:
    #   batch_size, seq_len = tf.shape(actions)[:2]
    #   states = self.initial(batch_size, seq_len)
    post, prior = self.obs_step(actions, embeds, is_first)
    return post, prior

  @tf.function
  def imagine(self, actions, state=None):
    '''
    input:
      actions:      a_t
      state.stoch:  z_t
    output:
      prior.stoch:  z_t\hat
      prior.deter:  h_t
      '''
    if state is None:
      state = self.initial(tf.shape(actions)[0])
    assert isinstance(state, dict), state
    batch_size, seq_len, _ = actions.shape
    while state['stoch'].shape[1]!=seq_len:
      prior = self.img_step(state['stoch'], actions)
      state['stoch'] = tf.concat([state['stoch'], tf.expand_dims(prior['stoch'][:,-1], 1)], 1)
    return self.img_step(state['stoch'], actions)

  def get_feat(self, state):
    stoch = self._cast(state['stoch'])
    if self._discrete:
      shape = stoch.shape[:-2] + [self._stoch * self._discrete]
      stoch = tf.reshape(stoch, shape)
    return tf.concat([stoch, state['deter']], -1)

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

  @tf.function
  def obs_step(self, actions, embeds, is_first, sample=True):
    '''
    input:
      embeds:       x_t
      actions:      a_t
    calculates:
      stochs:       z_t
    output:
      post.stoch:   z_t
      post.deter:   h_t
      prior.stoch:  z_t\hat
      prior.deter:  h_t
    '''
    if len(is_first.shape)<2: 
        is_first=is_first.reshape([1,1])
        embeds=embeds.reshape([1,1,-1])
    # Zero out firsts
    actions = tf.nest.map_structure(
        lambda x: tf.einsum('bl,bl...->bl...', 1.0 - is_first.astype(x.dtype), x), (actions))

    x = self.get('obs_out', tfkl.Dense, self._hidden)(embeds)
    x = self.get('obs_out_norm', common.NormLayer, self._norm)(x)
    stochs = self._act(x)  # z_t | x_t

    prior = self.img_step(stochs, actions, sample) # (z_t\hat | h_t), (h_t | z_t, a_t)

    stats = self._suff_stats_layer('obs_dist', x)
    dist = self.get_dist(stats)
    stochs = dist.sample() if sample else dist.mode()
    post = {'stoch': stochs, 'deter': prior['deter'], **stats} # (z_t | x_t), (h_t | z_t, a_t)
    return post, prior

  @tf.function
  def img_step(self, stochs, actions, sample=True):
    '''
    input:
      stochs:         z_t
      actions:        a_t 
    output:
      prior.stochs:   z_t\hat
      prior.deter:    h_t
    '''
    stochs = self._cast(stochs)
    actions = self._cast(actions)
    if self._discrete and stochs.shape[-1]!=self._stoch * self._discrete:
      shape = stochs.shape[:-2] + [self._stoch * self._discrete]
      stochs = tf.reshape(stochs, shape)
    batch_size, seq_len, _ = stochs.shape

    # with tf.GradientTape() as transformer_tape:
    x = tf.concat([stochs, actions[:,:seq_len]], -1) # z_t, a_t
    x = self.get('img_out', tfkl.Dense, self._hidden)(x)
    x = self.get('img_in_norm', common.NormLayer, self._norm)(x)
    token_embeds = self._act(x)
    # tokens = self.get('img_softargmax', common.Softargmax)(x)

    token_embeds = tf.concat([tf.zeros(shape=(batch_size, 1, self._hidden)), token_embeds], axis=1)
    # token_embeds = tf.keras.preprocessing.sequence.pad_sequences(token_embeds, maxlen=seq_len+1, padding='pre', dtype='float32')

    # with tf.device('cpu:0'):
    #   embedding_layer = self.get('in_embedding', tfkl.Embedding, input_dim=4096, output_dim=self._hidden, mask_zero=True)
    #   embedding_layer.build(input_shape=4096)
    # token_embeds = embedding_layer(tokens)
    input_embeds, target_embeds = token_embeds[:,:-1], token_embeds[:,1:]

    transformer = self.get('img_transformer', common.Transformer, num_layers=1, d_model=self._hidden, 
                            num_heads=1, dff=512, vocab_size=self._hidden, dropout_rate=0.1)

    deter = transformer((input_embeds, target_embeds),) # h_t | z_t, a_t
    loss_fn = tf.losses.CosineSimilarity(axis=-1)
    self.transformer_loss = loss_fn(target_embeds, deter)
      # print(f'{"Transformer loss:":25s} {transformer_loss.numpy()}')
    # self.transformer_opt(transformer_tape, self.transformer_loss, transformer)
    # pred_tokens = self.get('img_softargmax', common.Softargmax)(preds)
    # deter = self.get('in_embedding', tfkl.Embedding, input_dim=4096, output_dim=self._hidden, mask_zero=True)(pred_tokens)

    stats = self._suff_stats_ensemble(deter)
    index = tf.random.uniform((), 0, self._ensemble, tf.int32)
    stats = {k: v[index] for k, v in stats.items()}
    dist = self.get_dist(stats)
    stochs = dist.sample() if sample else dist.mode()
    prior = {'stoch': stochs, 'deter': deter, **stats}
    return prior

  def _suff_stats_ensemble(self, inp):
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
        k: v.reshape([v.shape[0]] + [len(inp)] + list(v.shape[2:]))
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