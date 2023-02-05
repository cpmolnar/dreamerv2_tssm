
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import dreamerv2.api as dv2
from dreamerv2.train_with_configs import main

logdir_path = r'C:\Users\Carl\OneDrive\Desktop'
task = 'dmc_walker_walk'
ssm_type = 'rssm_em'
exp_name = '_exp'

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

    'encoder.mlp_keys': '.*', 
    'encoder.cnn_keys': '$^',
    'decoder.mlp_keys': '.*', 
    'decoder.cnn_keys': '$^',
    'action_repeat': 2,
    'clip_rewards': 'identity',
    'pred_discount': False,
    'replay.prioritize_ends': False,
    'grad_heads': ['decoder', 'reward'],
    'rssm.hidden': 200, 
    'rssm.deter': 200,
    'model_opt.lr': 3e-4,
    'actor_opt.lr': 8e-5,
    'critic_opt.lr': 8e-5,
    'actor_ent': 1e-4,
    'kl.free': 1.0,

    'episodic_memory.max_size': 256,
    'episodic_memory.verbose': False,
    'dataset.batch': 8, 
    'dataset.length': 50,
    'load_model': True
}).parse_flags()

main(config)