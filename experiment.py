
import gym
import gym_minigrid
import dreamerv2.api as dv2

config = dv2.defaults.update({
    'jit': False,
    'time_limit': 100,
    'eval_every': 300,
    'log_every': 300,
    'prefill': 100,
    'pretrain': 1,
    'train_steps': 1,
    'logdir': '~/logdir/minigrid_tssm',
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'ssm_type': 'transformer'
}).parse_flags()

env = gym.make('MiniGrid-DoorKey-6x6-v0')
env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)