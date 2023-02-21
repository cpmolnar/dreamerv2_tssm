
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
    'jit': False,
    'task': task,
    'eval_every': 100,
    'log_every': 100,
    'prefill': 100,
    # 'logdir': f'{logdir_path}/logdir/{task}/{ssm_type}',
    'logdir': f'{logdir_path}/logdir/{task}/{ssm_type + exp_name}',
    'ssm_type': ssm_type,

    'episodic_memory.max_size': 128,
    'episodic_memory.verbose': False,
    # 'dataset.batch': 8, 
    # 'dataset.length': 50,
    'load_model': False
}).parse_flags()

main(config)