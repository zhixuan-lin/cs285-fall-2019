import subprocess
import glob
import os
import os.path as osp
import shlex
import argparse
from matplotlib import pyplot as plt
from cs285.experiments.utils.tb_reader import get_scalars

CS285_DIR = osp.dirname(osp.dirname(osp.realpath(__file__)))
DATA_DIR = osp.join(CS285_DIR, 'data')
def run_p1():
    commands = [
        'python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1',

        'python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1',

        'python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1',
    ]
    for command in commands:
        subprocess.run(shlex.split(command))

def run_p2():
    commands = [
        'python cs285/scripts/run_hw4_mb.py --exp_name obstacles_singleiteration --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10'
    ]
    for command in commands:
        subprocess.run(shlex.split(command))

def vis_p2():
    matched = glob.glob(osp.join(DATA_DIR, '*single*'))
    assert len(matched) == 1, 'p2 vis. {} matched'.format(len(matched))
    event_path = matched[0]
    df_train = get_scalars(event_path, 'Train_AverageReturn')
    df_eval = get_scalars(event_path, 'Eval_AverageReturn')
    assert df_train.shape[0] == df_eval.shape[0] == 1
    _, ax = plt.subplots()
    ax.scatter([1], df_train['value'], label='Train')
    ax.scatter([1], df_eval['value'], label='Eval')
    ax.set_xlabel('step')
    ax.set_ylabel('return')
    ax.set_title('Eval vs Train Return')
    ax.legend()
    plt.show()


def run_p3():
    commands = [
        'python cs285/scripts/run_hw4_mb.py --exp_name obstacles --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12',

        'python cs285/scripts/run_hw4_mb.py --exp_name reacher --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name cheetah --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20'
    ]
    for command in commands:
        subprocess.run(shlex.split(command))

def vis_p3():
    patterns = {
        'obstacles': '*obstacles_obstacles*',
        'cheetah': '*cheetah_cheetah*',
        'reacher': '*reacher_reacher*',
    }
    matched = {k: glob.glob(osp.join(DATA_DIR, patterns[k])) for k in patterns}
    assert all(len(matched[k]) == 1 for k in matched)
    event_files = {k: v[0] for k, v in matched.items()}
    for name, event_file in event_files.items():
        df = get_scalars(event_file, 'Eval_AverageReturn')
        _, ax = plt.subplots()
        ax.plot(df['step'], df['value'])
        ax.set_title(name)
    plt.show()


def run_p4():
    commands = [
        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon5 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon15 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon30 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq100 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 ',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq1000 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 1000',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble1 --env_name reacher-cs285-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble3 --env_name reacher-cs285-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

        'python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble5 --env_name reacher-cs285-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15',

    ]
    for command in commands:
        subprocess.run(shlex.split(command))

def vis_p4():
    patterns = {
        'horizon':
        {
            5: '*horizon5*',
            15: '*horizon15*',
            30: '*horizon30*',
        },
        'numseq':
        {
            100: '*numseq100_*',
            1000: '*numseq1000_*',
        },
        'ensemble':
        {
            1: '*ensemble1*',
            3: '*ensemble3*',
            5: '*ensemble5*',
        }
    }
    def apply(item, op):
        if isinstance(item, dict):
            for k in item:
                item[k] = apply(item[k], op)
            return item
        else:
            return op(item)
    op = lambda x: glob.glob(osp.join(DATA_DIR, x))
    matched = apply(patterns, op)
    def check(x):
        assert len(x) == 1, x
        return x
    apply(matched, check)
    event_files = apply(matched, lambda x: x[0])
    for name, values in event_files.items():
        _, ax = plt.subplots()
        for value, event_path in values.items():
            df = get_scalars(event_path, 'Eval_AverageReturn')
            ax.plot(df['step'], df['value'], label=value)
        ax.set_title(name)
        ax.set_xlabel('step')
        ax.set_xlabel('return')
        ax.legend()

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    args = parser.parse_args()
    exp = {
        'run-p1': run_p1,
        'run-p2': run_p2,
        'vis-p2': vis_p2,
        'run-p3': run_p3,
        'vis-p3': vis_p3,
        'run-p4': run_p4,
        'vis-p4': vis_p4,
    }
    assert args.exp in exp
    exp[args.exp]()
