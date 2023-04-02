import tensorflow as tf
from tensorflow.keras import mixed_precision as prec

import common
import expl


class Agent(common.Module):

  def __init__(self, config, obs_space, act_space, step):
    self.config = config
    self.obs_space = obs_space
    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = WorldModel(config, obs_space, self.tfstep)
    d_k = d_v = config.rssm['hidden'] + (config.rssm['stoch'] * config.rssm['discrete'])
    self.em = common.RaggedEpisodicMemory(config.dataset['length'], d_k, d_v, **config.episodic_memory)
    self._task_behavior = ActorCritic(config, self.act_space, self.tfstep)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action, None
    latent, action, prev_skill = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    latent_exp = {k: tf.expand_dims(v, 0) for k, v in latent.items()}
    feat = self.wm.rssm.get_feat(latent_exp)
    # if prev_skill is not None:
    #   skill_rew = tf.map_fn(lambda i: max_cosine_sim(prev_skill[:,i], feat[:,i])[0, 0], tf.range(feat.shape[1]), fn_output_signature=tf.float32)[0]
    #   skill = tf.where(tf.expand_dims(tf.math.logical_or((skill_rew==0), (skill_rew>0.9)), -1), self.em.retrieve_skill(feat), prev_skill)
    # else:
    skill = self.em.retrieve_skill(feat)
    input = {**latent_exp, 'skill': skill, 'feat': feat}
    if mode == 'eval':
      actor = self._task_behavior.actor(input)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(input)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(input)
      action = actor.sample()
      noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)[0]
    outputs = {'action': action}
    state = (latent, action, skill)
    return outputs, state

  @tf.function
  def train(self, data, state=None, update_mem=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    if update_mem:
      keys = self.wm.rssm.get_feat(start)
      values = data['reward'][:,:,None]
      self.em.cleanup()
      self.em.add_seqs(keys, values)
    reward = lambda seq: self.wm.heads['reward'](seq['feat']).mode()
    metrics.update(self._task_behavior.train(
        self.wm, self.em, start, data['is_terminal'], reward))
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  @tf.function
  def report(self, data):
    report = {}
    data = self.wm.preprocess(data)
    for key in self.wm.heads['decoder'].cnn_keys:
      name = key.replace('/', '_')
      report[f'openl_{name}'] = self.wm.video_pred(data, key, self.em)
    return report


class WorldModel(common.Module):

  def __init__(self, config, obs_space, tfstep):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items() if v.shape is not None}
    self.config = config
    self.tfstep = tfstep
    self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.heads = {}
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
    self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    for name in config.grad_heads:
      assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)

  def train(self, data, state=None, em=None):
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state, em)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None, em=None):
    data = self.preprocess(data)
    embed = self.encoder(data) # z_t
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state) # out: z_t\hat, z_t. in: z_t, a_t
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    state_feat = self.rssm.get_feat(state)
    skill = em.retrieve_skill(state_feat)
    feat = self.rssm.get_feat(prior)
    data['reward'] -= tf.norm(skill - feat, axis=-1)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post) # z_t\hat, h_t
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, em, start, is_terminal, horizon):
    # flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    # start = {k: flatten(v) for k, v in start.items()}
    feat = self.rssm.get_feat(start)
    skill = em.retrieve_skill(feat)
    start['action'] = tf.zeros_like(policy({**start, 'skill': skill, 'feat': feat}).mode())
    seq = {k: [v] for k, v in start.items()}
    seq['feat'] = [feat]
    seq['skill'] = [skill]
    seq['skill_rew'] = [tf.zeros(shape=(skill.shape[:2]))]
    for _ in range(horizon):
      input = {k: v[-1] for k, v in seq.items() if k in ['logit', 'stoch', 'deter', 'skill', 'feat']}
      action = policy(sg(input)).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      # skill_rew = tf.map_fn(lambda i: max_cosine_sim(input['skill'][:,i], feat[:,i])[0, 0], tf.range(feat.shape[1]), fn_output_signature=tf.float32)[None]
      skill_rew = -tf.norm(input['skill'] - feat, axis=-1)
      # skill = tf.where(tf.expand_dims(tf.math.logical_or((skill_rew==0), (skill_rew>0.9)), -1), em.retrieve_skill(feat), input['skill'])
      skill = em.retrieve_skill(feat)
      for key, value in {**state, 'action': action, 'skill': skill, 'feat': feat, 'skill_rew': skill_rew}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - is_terminal.astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key, em):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data) # x_t
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5]) # get z_t, h_t
    start_feat = self.rssm.get_feat(states)
    recon = decoder(start_feat)[key].mode()[:6] # get x_t\hat
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    imag_feat = self.rssm.get_feat(prior)
    openl = decoder(imag_feat)[key].mode()

    feat = tf.concat([start_feat, imag_feat], 1)
    skill = em.retrieve_skill(feat)
    goal_model = decoder(skill)[key].mode() + 0.5

    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error, goal_model], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))


class ActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep):
    self.config = config
    self.act_space = act_space
    self.tfstep = tfstep
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.PolicyHierarchy(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)

  def train(self, wm, em, start, is_terminal, reward_fn):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      idx = tf.random.uniform((), minval=0, maxval=self.config.dataset.batch, dtype=tf.int32)
      seq = wm.imagine(self.actor, em, {k: v[None, idx] for k, v in start.items()}, is_terminal[None, idx], hor)
      reward = reward_fn(seq)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)
    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4)
    metrics.update({'skill_rew_mean': seq['skill_rew'].mean()})
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
    flatten = lambda x: tf.squeeze(x.reshape([x.shape[0]] + [x.shape[1] * x.shape[2]] + [-1]))
    target = flatten(target)
    seq = {k: flatten(v) for k, v in seq.items()}
    input = {k: v[:-2] for k, v in seq.items() if k in ['logit', 'stoch', 'deter', 'skill', 'feat']}
    policy = self.actor(sg(input))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)

def max_cosine_sim(x, y):
  if tf.norm(x, axis=-1)==0 or tf.norm(y, axis=-1)==0: 
    return tf.zeros(shape=(x.shape[0], y.shape[0]))
  m = tf.maximum(tf.norm(x, axis=-1), tf.norm(y, axis=-1))
  return (x/m) @ tf.transpose(y/m)