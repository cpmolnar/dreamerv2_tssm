
import gymnasium as gym
import minigrid
import dreamerv2.api as dv2

task = 'gym_minigrid'
ssm_type = 'tssm'
assert ssm_type in ['tssm', 'rssm'], 'invalid ssm_type'

config = dv2.defaults.update({
    'task': task,
    'jit': False,
    'time_limit': 100,
    'eval_every': 300,
    'log_every': 300,
    'prefill': 100,
    'pretrain': 1,
    'train_steps': 1,
    'logdir': f'~/logdir/{task}/{ssm_type}',
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'ssm_type': ssm_type
}).parse_flags()

# import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(not config.jit)
# tf.data.experimental.enable_debug_mode()
# message = 'No GPU found. To actually train on CPU remove this assert.'
# assert tf.config.experimental.list_physical_devices('GPU'), message
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make("MiniGrid-DoorKey-6x6-v0")
env = minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)