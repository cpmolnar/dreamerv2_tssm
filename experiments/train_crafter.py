
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import dreamerv2.api as dv2
from dreamerv2.train_with_configs import main

# logdir_path = r'/home/paperspace/dreamerv2_tssm/dreamerv2'
logdir_path = r'C:/Users/Carl/OneDrive/Desktop'
task = 'crafter_reward'
ssm_type = 'rssm_em'
exp_name = '_pf100'

config = dv2.defaults.update({
    'task': task,
    'jit': False,
    'time_limit': 100,
    'eval_every': 100,
    'log_every': 100,
    'prefill': 100,
    'pretrain': 1,
    'train_steps': 1,
    # 'logdir': f'{logdir_path}/logdir/{task}/{ssm_type}',
    'logdir': f'{logdir_path}/logdir/{task}/{ssm_type + exp_name}',
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
    # '.*\.norm': 'layer',
    'expl_behavior': 'greedy',#'Plan2Explore'


    'episodic_memory.max_size': 1024,
    # 'episodic_memory.verbose': True,
    'dataset.batch': 8, 
    'dataset.length': 50,
    'load_model': True
}).parse_flags()

main(config)