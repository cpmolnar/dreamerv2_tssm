
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import dreamerv2.api as dv2
from dreamerv2.train_with_configs import main

task = 'crafter_reward'
ssm_type = 'rssm'

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
    'ssm_type': ssm_type,

    'encoder.mlp_keys': '$^', 
    'encoder.cnn_keys': 'image',
    'decoder.mlp_keys': '$^', 
    'decoder.cnn_keys': 'image',
    'log_keys_max': '^log_achievement_.*',
    'log_keys_sum': '^log_reward$',
    'rssm.hidden': 1024,
    'rssm.deter': 1024,
    'discount': 0.999,
    'model_opt.lr': 1e-4,
    'actor_opt.lr': 1e-4,
    'critic_opt.lr': 1e-4,
    '.*\.norm': 'layer',
}).parse_flags()

# assert config.action_repeat == 1

# suite, task = config.task.split('_', 1)
# reward = bool(['noreward', 'reward'].index(task))
# env = common.Crafter(config.logdir, reward)
# env = common.OneHotAction(env)

main(config)