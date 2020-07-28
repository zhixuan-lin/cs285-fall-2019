import argparse
import pandas as pd
import random
import json
import os
from cycler import cycler
import sys
import os.path as osp
import subprocess
import shlex
import time
import numpy as np
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import glob
from cs285.experiments.utils.tb_reader import get_scalars
from cs285.experiments.utils.multicmd import run_multiple_commands
from cs285.experiments.utils.plotstd import plot_mean_std


EXP_DIR = osp.dirname(osp.realpath(__file__))
CS285_DIR = osp.dirname(EXP_DIR)
ROOT_DIR = osp.dirname(CS285_DIR)
LOG_DIR = osp.join(ROOT_DIR, 'run_logs')


def run_q1():
    command = f'python {CS285_DIR}/scripts/run_hw3_dqn.py --env_name PongNoFrameskip-v4 --exp_name q1'
    command = shlex.split(command)
    subprocess.run(command)

def vis_q1():
    path = 'dqn_q1_PongNoFrameskip-v4*'
    matched = glob.glob(osp.join(LOG_DIR, path))
    assert len(matched) == 1, '{} matches found, expected {}'.format(len(matched), 1)
    logpath = matched[0]
    avg_return = get_scalars(logpath, 'Train_AverageReturn')
    best_return = get_scalars(logpath, 'Train_BestReturn')
    _, ax = plt.subplots()
    ax.plot(avg_return['step'], avg_return['value'], label='average return')
    ax.plot(best_return['step'], best_return['value'], label='best return')
    ax.set_xlabel('step')
    ax.set_ylabel('return')
    ax.set_title('DQN-Pong')
    ax.legend()
    plt.show()

def run_q2():
    dqn_commands = [
        f'python {CS285_DIR}/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_dqn_{seed} --seed {seed}'
        for seed in [1, 2, 3]
    ]
    ddqn_commands = [
        f'python {CS285_DIR}/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q2_doubledqn_{seed} --double_q --seed {seed}'
        for seed in [1, 2, 3]
    ]
    commands = dqn_commands + ddqn_commands
    logdir = osp.join(CS285_DIR, 'data', 'logs')
    os.makedirs(logdir, exist_ok=True)
    logpaths = [osp.join(logdir, f'{i}.log') for i in range(len(commands))]
    metalogpath = osp.join(logdir, 'meta.log')
    run_multiple_commands(commands, logpaths, metalogpath)

def vis_q2():
    dqn_paths = glob.glob(osp.join(LOG_DIR, '*q2_dqn*'))
    ddqn_paths = glob.glob(osp.join(LOG_DIR, '*q2_doubledqn*'))
    assert len(dqn_paths) == 3, 'DQN: {} matches found, expected {}'.format(len(dqn_paths), 3)
    assert len(ddqn_paths) == 3, 'DDQN: {} matches found, expected {}'.format(len(ddqn_paths), 3)

    dqn_data = [get_scalars(path, 'Train_AverageReturn') for path in dqn_paths]
    ddqn_data = [get_scalars(path, 'Train_AverageReturn') for path in ddqn_paths]

    dqn_mean = np.mean(list(map(lambda x: x['value'], dqn_data)), axis=0)
    dqn_std = np.std(list(map(lambda x: x['value'], dqn_data)), axis=0)

    ddqn_mean = np.mean(list(map(lambda x: x['value'], ddqn_data)), axis=0)
    ddqn_std = np.std(list(map(lambda x: x['value'], ddqn_data)), axis=0)

    _, ax = plt.subplots()
    cmap = mpl.cm.get_cmap('Accent', lut=2)
    colors = cycler('c', cmap(np.linspace(0, 1, 2)))

    for (mean, std, style, label) in zip([dqn_mean, ddqn_mean], [dqn_std, ddqn_std], colors, ['DQN', 'DDQN']):
        plot_mean_std(ax, mean, std, x=dqn_data[0]['step'], label=label, **style)

    ax.set_xlabel('step')
    ax.set_ylabel('return')
    ax.set_title('DQN vs DDQN')

    ax.legend()
    plt.show()

UPDATES = [3000, 1000, 2000, 5000, 10000]

def run_q3():
    commands = [
        f'python {CS285_DIR}/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q3_hparam{i}_update{update} --target_update_freq {update}'.format(i=i+1, update=update)
        for i, update in enumerate(UPDATES)
        # for i, update in enumerate([10000])
    ]

    logdir = osp.join(CS285_DIR, 'data', 'logs')
    os.makedirs(logdir, exist_ok=True)
    logpaths = [osp.join(logdir, f'{i}.log') for i in range(len(commands))]
    metalogpath = osp.join(logdir, 'meta.log')
    run_multiple_commands(commands, logpaths, metalogpath)

def vis_q3():
    logpaths = {}
    for update in UPDATES:
        matched = glob.glob(osp.join(LOG_DIR, f'*q3*update{update}_*'))
        assert len(matched) == 1
        logpaths[update] = matched[0]
    data = {update: get_scalars(logpaths[update], 'Train_AverageReturn') for update in UPDATES}

    _, ax = plt.subplots()
    cmap = mpl.cm.get_cmap('Accent', len(UPDATES))
    styles = cycler('c', cmap(np.linspace(0, 1, len(UPDATES))))
    for update, style in zip(UPDATES, styles):
        ax.plot(data[update]['step'], data[update]['value'], **style, label=update)

    ax.legend(title='target_update_freq')
    plt.show()

Q4_CONFIGS = [
    (1, 1),
    (1, 100),
    (100, 1),
    (10, 10),
]

def run_q4():
    commands = [
        f'python {CS285_DIR}/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name {ntu}_{ngsptu}, -ntu {ntu} -ngsptu {ngsptu}' for (ntu, ngsptu) in Q4_CONFIGS
    ]

    logdir = osp.join(CS285_DIR, 'data', 'logs')
    os.makedirs(logdir, exist_ok=True)
    logpaths = [osp.join(logdir, f'{i}.log') for i in range(len(commands))]
    metalogpath = osp.join(logdir, 'meta.log')
    run_multiple_commands(commands, logpaths, metalogpath)

def vis_q4():
    logpaths = {}
    for config in Q4_CONFIGS:
        matched = glob.glob(osp.join(LOG_DIR, 'ac_{}_{}_CartPole*'.format(*config)))
        assert len(matched) == 1
        logpaths[config] = matched[0]
    data = {config: get_scalars(logpaths[config], 'Eval_AverageReturn') for config in Q4_CONFIGS}

    _, ax = plt.subplots()
    cmap = mpl.cm.get_cmap('Accent', len(Q4_CONFIGS))
    styles = cycler('c', cmap(np.linspace(0, 1, len(Q4_CONFIGS))))
    for update, style in zip(Q4_CONFIGS, styles):
        ax.plot(data[update]['step'], data[update]['value'], **style, label=update)

    ax.legend(title='ntu ngsptu')
    plt.show()
    

def run_q5():
    commands = [
        'python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10',
        'python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10'
    ]

    logdir = osp.join(CS285_DIR, 'data', 'logs')
    os.makedirs(logdir, exist_ok=True)
    logpaths = [osp.join(logdir, f'{i}.log') for i in range(len(commands))]
    metalogpath = osp.join(logdir, 'meta.log')
    run_multiple_commands(commands, logpaths, metalogpath)


Q5_ENVS = ['InvertedPendulum', 'HalfCheetah']
def vis_q5():
    logpaths = {}
    for env in Q5_ENVS:
        matched = glob.glob(osp.join(LOG_DIR, f'*{env}*'))
        assert len(matched) == 1
        logpaths[env] = matched[0]
    data = {env: get_scalars(logpaths[env], 'Eval_AverageReturn') for env in Q5_ENVS}

    _, axes = plt.subplots(ncols=2)
    for i, env in enumerate(Q5_ENVS):
        axes[i].plot(data[env]['step'], data[env]['value'], label=env)

    [ax.legend() for ax in axes]
    plt.show()


# def plot_mean_std(ax, means, stds=None, x=None, label=None, c='y', linestyle='-'):
#     sns.set(style="darkgrid")
#     # fig, ax = plt.subplots()
#     means = np.array(means)
#     stds = np.array(stds) if stds is not None else stds
#     x = np.array(x) if x is not None else x


#     if x is not None:
#         ax.plot(x, means, label=label, c=c, linestyle=linestyle)
#         if stds is not None:
#             ax.fill_between(x, means-stds, means+stds, alpha=0.3, facecolor=c)
#     else:
#         ax.plot(means, label=label, c=c, linestyle=linestyle)
#         if stds is not None:
#             ax.fill_between(np.arange(len(means)), means-stds, means+stds, alpha=0.3, facecolor=c)

#     ax.legend()


# def run_multiple_commands(commands, logpaths, metalogpath):
#     import subprocess
#     import shlex
#     import time
#     assert len(commands) == len(logpaths)
#     logfiles = []
#     processes = []
#     try:
#         metalogfile = open(metalogpath, 'w', buffering=1)
#         for command, logpath in zip(commands, logpaths):
#             logfile = open(logpath, 'w', buffering=1)
#             logfiles.append(logfile)
#             p = subprocess.Popen(shlex.split(command), stdout=logfile)
#             processes.append(p)
#             print(f'{command}\n=> {logpath}', file=metalogfile)

#         while len(processes) > 0:
#             terminated = []
#             for i in range(len(processes)):
#                 return_code = processes[i].poll()
#                 if return_code is not None:
#                     terminated.append(i)
#             for i in reversed(terminated):
#                 print('TERMINATED: {}'.format(commands[i]), file=metalogfile)
#                 del processes[i]
#                 del commands[i]
#                 del logpaths[i]
#                 logfiles[i].close()
#                 del logfiles[i]
#             time.sleep(5)
#         print('All processes have terminated. Quitting...')
#     finally:
#         for file in logfiles:
#             file.close()


def run_p3():
    seeds = range(5)
    templates = [
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name sb_no_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name sb_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name sb_rtg_na_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name lb_no_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name lb_rtg_dsa_seed_{seed} --seed {seed}",
        "python cs285/scripts/run_hw2_policy_gradient.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name lb_rtg_na_seed_{seed} --seed {seed}",
    ]
    commands = []
    for template in templates:
        for seed in seeds:
            command = template.format(seed=seed)
            commands.append(command)
    os.makedirs('cs285/scripts/logs', exist_ok=True)
    logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
    # print(len(commands))
    metalogfile = 'cs285/experiments/logs/meta.log'
    run_multiple_commands(commands, logfiles, metalogfile)
    
def parse_p4(name):
    tokens = name.split('_')
    bs = int(tokens[2][1:])
    lr = float(tokens[3][1:])
    return lr, bs

def get_p4_tested():
    results = []
    for logdir in os.listdir('cs285/data'):
        if 'pg_ip' in logdir:
            lr, bs = parse_p4(logdir)
            results.append((lr, bs))
    
    return results
    
P4_SEARCH = False
def run_p4():
    if P4_SEARCH:
        lrs = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
        bss = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        num = 40

        tested = get_p4_tested()
        combinations = [(lr, bs) for lr in lrs for bs in bss if (lr, bs) not in tested]
        
        assert num <= len(combinations)

        template = "python cs285/scripts/run_hw2_policy_gradient.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {bs} -lr {lr} -rtg --exp_name ip_b{bs}_r{lr}"
        configs_indices = np.random.choice(len(combinations), num, replace=False)
        configs = [combinations[i] for i in configs_indices]
        commands = []
        for lr, bs in configs:
            command = template.format(lr=lr, bs=bs)
            commands.append(command)
        os.makedirs('cs285/scripts/logs', exist_ok=True)
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        # print(len(commands))
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)
    else:
        seeds = [0]
        template = "python cs285/scripts/run_hw2_policy_gradient.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b {bs} -lr {lr} -rtg --exp_name ip_b{bs}_r{lr}_s{seed} --seed {seed}"
        lr = 2e-2
        bs = 300
        commands = []
        for seed in seeds:
            command = template.format(lr=lr, bs=bs, seed=seed)
            commands.append(command)
        subprocess.run(shlex.split(commands[0]))
        # logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        # print(len(commands))
        # metalogfile = 'cs285/experiments/logs/meta.log'
        # run_multiple_commands(commands, logfiles, metalogfile)
        
    
def vis_p4():
    if P4_SEARCH:
        template = 'cs285/data/pg_ip_b{bs}_r{lr}_InvertedPendulum-v2'
        metric_template = 'cs285/data/pg_ip_b{bs}_r{lr}_InvertedPendulum-v2/metrics_{i}.json'
        
        threshold = 900
        
        data = defaultdict(list)
        for lr, bs in get_p4_tested():
            logdir = template.format(bs=bs, lr=lr)
            if osp.isdir(logdir):
                for i in range(100):
                    with open(metric_template.format(lr=lr, bs=bs, i=i)) as f:
                        metrics = json.load(f)
                        data[(lr, bs)].append(metrics['Eval_AverageReturn'])
                        
        count = {}
        for key, value in data.items():
            count[key] = (np.array(value) > threshold).sum()
            
        with open('cs285/data/searched.pkl', 'wb') as f:
            pickle.dump(count, f)
            
        plot_hyper(count)
    else:
        metric_template = 'cs285/data/pg_ip_b{bs}_r{lr}_s{seed}_InvertedPendulum-v2/metrics_{i}.json'
        
        seeds = [0]

        lr = 2e-2
        bs = 300
        data = []
        for seed in seeds:
            data_thisseed = []
            for i in range(100):
                with open(metric_template.format(lr=lr, bs=bs, seed=seed, i=i)) as f:
                    metrics = json.load(f)
                    data_thisseed.append(metrics['Eval_AverageReturn'])
            data.append(data_thisseed)
        mean = np.mean(data, axis=0)
        # std = np.std(data, axis=0)
        f, ax = plt.subplots()
        plot_mean_std(ax, mean, stds=None, label='lr={lr},bs={bs}'.format(lr=lr, bs=bs))
        f.savefig('pdf/figures/p4-final.png')
        plt.show()
        
def run_p6():
    command = 'python cs285/scripts/run_hw2_policy_gradient.py --env_name LunarLanderContinuous-v2 --ep_len 1000 --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005 -eb 5000'
    subprocess.run(shlex.split(command))
    
def vis_p6():
    metric_template = 'cs285/data/pg_ll_b40000_r0.005_LunarLanderContinuous-v2/metrics_{i}.json'
    ret_list = []
    std_list = []
    for i in range(100):
        metric_path = metric_template.format(i=i)
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        ret = metrics['Eval_AverageReturn']
        std = metrics['Eval_StdReturn']
        ret_list.append(ret)
        std_list.append(std)
        
    f, ax = plt.subplots()
    plot_mean_std(ax, ret_list, std_list)
    plt.savefig('pdf/figures/p6.png')
    plt.show()

P7_SEARCH = False

def run_p7():
    if P7_SEARCH:
        lrs = [0.005, 0.01, 0.02]
        bss = [10000, 30000, 50000]
        
        
        template = 'python cs285/scripts/run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1 --reward_to_go --nn_baseline --exp_name hc_b{bs}_lr{lr}_nnbaseline'
        commands = []
        for lr in lrs:
            for bs in bss:
                command = template.format(lr=lr, bs=bs)
                commands.append(command)
        os.makedirs('cs285/scripts/logs', exist_ok=True)
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        # print(len(commands))
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)
    else:
        lr = 0.02
        bs = 30000
        commands = [
            'python cs285/scripts/run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1 --reward_to_go --nn_baseline --exp_name hc_b{bs}_lr{lr}_rtg_baseline',
            'python cs285/scripts/run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1 --reward_to_go               --exp_name hc_b{bs}_lr{lr}_rtg_nobaseline',
            'python cs285/scripts/run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1                --nn_baseline --exp_name hc_b{bs}_lr{lr}_nortg_baseline',
            'python cs285/scripts/run_hw2_policy_gradient.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {bs} -lr {lr} --video_log_freq -1                              --exp_name hc_b{bs}_lr{lr}_nortg_nobaseline'
        ]
        commands = [c.format(lr=lr, bs=bs) for c in commands]
        logfiles = [f'cs285/experiments/logs/{i}.log' for i in range(len(commands))]
        # print(len(commands))
        metalogfile = 'cs285/experiments/logs/meta.log'
        run_multiple_commands(commands, logfiles, metalogfile)
    
def vis_p7():
    if P7_SEARCH:
        metric_template = 'cs285/data/pg_hc_b{bs}_lr{lr}_nnbaseline_HalfCheetah-v2/metrics_{i}.json'
        name_template = 'b{bs}_lr{lr}'
        lrs = [0.005, 0.01, 0.02]
        bss = [10000, 30000, 50000]

        cycle = (cycler('color', ['r', 'g', 'b']) * cycler('linestyle', ['-', '--', '-.']))
        
        mean_list = defaultdict(list)
        std_list = defaultdict(list)
        f, ax = plt.subplots()
        for lr in lrs:
            for bs in bss:
                name = name_template.format(lr=lr, bs=bs)
                for i in range(100):
                    metric_file = metric_template.format(lr=lr, bs=bs, i=i)
                    with open(metric_file, 'r') as f:
                        metrics = json.load(f)
                    mean_list[name].append(metrics['Eval_AverageReturn'])
                    std_list[name].append(metrics['Eval_StdReturn'])

        
        for name, style in zip(mean_list, cycle):
            ax.plot(mean_list[name], label=name)
        ax.legend()
            # plot_mean_std(ax, mean_list[name], stds=None, label=name, c=style['color'], linestyle=style['linestyle'])
        plt.savefig('pdf/figures/p7-hyper.png')
        plt.show()
    else:
        logpaths = [
            'cs285/data/pg_hc_b{bs}_lr{lr}_rtg_baseline_HalfCheetah-v2/metrics_{i}.json',
            'cs285/data/pg_hc_b{bs}_lr{lr}_rtg_nobaseline_HalfCheetah-v2/metrics_{i}.json',
            'cs285/data/pg_hc_b{bs}_lr{lr}_nortg_baseline_HalfCheetah-v2/metrics_{i}.json',
            'cs285/data/pg_hc_b{bs}_lr{lr}_nortg_nobaseline_HalfCheetah-v2/metrics_{i}.json'
        ]
        names = [
            'rtg_baseline',
            'rtg_nobaseline',
            'nortg_baseline',
            'nortg_nobaseline',
        ]
        lr = 0.02
        bs = 30000
        data = defaultdict(list)
        for name, pathtem in zip(names, logpaths):
            for i in range(100):
                path = pathtem.format(lr=lr, bs=bs, i=i)
                with open(path, 'r') as f:
                    metrics = json.load(f)
                data[name].append(metrics['Eval_AverageReturn'])
                
        for name in names:
            plt.plot(data[name], label=name)
        plt.legend()
        plt.savefig('pdf/figures/p7-final.png')
        plt.show()
            
    
    
    
        
    
def plot_hyper(count):
    import pandas as pd
    datapoints = [(lr, bs, c) for (lr, bs), c in count.items()]
    lrs, bss, css = zip(*datapoints)
    df = pd.DataFrame(dict(lr=(lrs), bs=bss, c=css))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_xlim((1e-3, 1.0))
    sns.scatterplot(x='lr', y='bs', size='c', data=df, sizes=(10, 100))
    plt.savefig('pdf/figures/p4-hyper.png')
    plt.show()
    
    
    
def vis_p3():
    exp_names = [
        "sb_no_rtg_dsa",
        "sb_rtg_dsa",
        "sb_rtg_na",
        "lb_no_rtg_dsa",
        "lb_rtg_dsa",
        "lb_rtg_na",
    ]
    seeds = list(range(5))
    data = defaultdict(dict)
    template = 'cs285/data/pg_{exp_name}_seed_{seed}_CartPole-v0/metrics_{step}.json'
    
    for exp_name in exp_names:
        data_diff_seeds = []
        for seed in seeds:
            data_this_seed = []
            for step in range(100):
                metric_path = template.format(exp_name=exp_name, seed=seed, step=step)
                with open(metric_path, 'r') as f:
                    metrics = json.load(f)
                    data_this_seed.append(metrics['Eval_AverageReturn'])
            data_diff_seeds.append(data_this_seed)
        data[exp_name]['raw_data'] = data_diff_seeds
        data[exp_name]['return_mean'] = np.mean(data_diff_seeds, axis=0)
        data[exp_name]['return_std'] = np.std(data_diff_seeds, axis=0)

    f, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5))
    cycle = (cycler('color', ['r', 'g', 'b', 'y', 'm', 'c']) + cycler('linestyle', ['-'] * 6))
    for (exp_name, style) in zip(exp_names, cycle):
        if 'sb' in exp_name:
            ax = axes[0]
        else:
            ax = axes[1]
        means = data[exp_name]['return_mean']
        stds = data[exp_name]['return_std']
        plot_mean_std(ax, means, stds=None, label=exp_name, c=style['color'], linestyle=style['linestyle'])
    if not osp.isdir('pdf/figures'):
        os.mkdir('pdf/figures')
    plt.savefig('pdf/figures/p3.png')
    plt.show()
        # print('exp_name')
        # print(data[exp_name]['return_mean'])
        # print(data[exp_name]['return_std'])
        
        
        
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    args = parser.parse_args()
    
    exp = {
        'run-q1': run_q1,
        'vis-q1': vis_q1,
        'run-q2': run_q2,
        'vis-q2': vis_q2,
        'run-q3': run_q3,
        'vis-q3': vis_q3,
        'run-q4': run_q4,
        'vis-q4': vis_q4,
        'run-q5': run_q5,
        'vis-q5': vis_q5,

        'run-p4': run_p4,
        'run-p6': run_p6,
        'run-p7': run_p7,
        'vis-p3': vis_p3,
        'vis-p4': vis_p4,
        'vis-p6': vis_p6,
        'vis-p7': vis_p7,
    }
    assert args.exp in exp

    sns.set(style="darkgrid")
    exp[args.exp]()


if __name__ == '__main__':
    main()
