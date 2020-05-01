import argparse
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
import pickle

def plot_mean_std(ax, means, stds=None, x=None, label=None, c='y', linestyle='-'):
    sns.set(style="darkgrid")
    # fig, ax = plt.subplots()
    means = np.array(means)
    stds = np.array(stds) if stds is not None else stds
    x = np.array(x) if x is not None else x


    if x is not None:
        ax.plot(x, means, label=label, c=c, linestyle=linestyle)
        if stds is not None:
            ax.fill_between(x, means-stds, means+stds, alpha=0.3, facecolor=c)
    else:
        ax.plot(means, label=label, c=c, linestyle=linestyle)
        if stds is not None:
            ax.fill_between(np.arange(len(means)), means-stds, means+stds, alpha=0.3, facecolor=c)

    ax.legend()


def run_multiple_commands(commands, logpaths, metalogpath):
    import subprocess
    import shlex
    import time
    assert len(commands) == len(logpaths)
    logfiles = []
    processes = []
    try:
        metalogfile = open(metalogpath, 'w', buffering=1)
        for command, logpath in zip(commands, logpaths):
            logfile = open(logpath, 'w', buffering=1)
            logfiles.append(logfile)
            p = subprocess.Popen(shlex.split(command), stdout=logfile)
            processes.append(p)
            print(f'{command}\n=> {logpath}', file=metalogfile)

        while len(processes) > 0:
            terminated = []
            for i in range(len(processes)):
                return_code = processes[i].poll()
                if return_code is not None:
                    terminated.append(i)
            for i in reversed(terminated):
                print('TERMINATED: {}'.format(commands[i]), file=metalogfile)
                del processes[i]
                del commands[i]
                del logpaths[i]
                logfiles[i].close()
                del logfiles[i]
            time.sleep(5)
        print('All processes have terminated. Quitting...')
    finally:
        for file in logfiles:
            file.close()


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
    assert os.path.isdir('cs285/scripts'), 'Please run this from hw2 root'
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    args = parser.parse_args()
    
    exp = {
        'run-p3': run_p3,
        'run-p4': run_p4,
        'run-p6': run_p6,
        'run-p7': run_p7,
        'vis-p3': vis_p3,
        'vis-p4': vis_p4,
        'vis-p6': vis_p6,
        'vis-p7': vis_p7,
    }
    assert args.exp in exp
    exp[args.exp]()


if __name__ == '__main__':
    main()
