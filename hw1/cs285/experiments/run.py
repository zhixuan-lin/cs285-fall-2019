import shlex, subprocess
import numpy as np
import sys
import os
import argparse
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('exp')
args = parser.parse_args()



def run_1_2():
	os.system('python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --eval_batch_size 50000 --num_agent_train_steps_per_iter 10000')
	os.system('python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 --expert_data cs285/expert_data/expert_data_Hopper-v2.pkl --eval_batch_size 50000 --num_agent_train_steps_per_iter 10000')

def run_1_3():
	seeds = range(10)
	template = "python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name hyper_ant_{num_iter}_seed_{seed} --n_iter 1 --expert_data cs285/expert_data/expert_data_Ant-v2.pkl --eval_batch_size 5000 --num_agent_train_steps_per_iter {num_iter} --seed {seed}"
	iters = range(1000, 10000+1, 1000)
	for seed in seeds:
		for i in iters:
			command = template.format(num_iter=i, seed=seed)
			os.system(command)

def vis_1_3():
	iters = range(1000, 10000+1, 1000)
	seeds = range(10)
	name = [f'hyper_ant_{i}' for i in iters]
	folders = defaultdict(list)
	for folder in glob.glob('cs285/data/*'):
		if 'hyper_ant' in folder:
			tokens = folder.split('_')
			index = int(tokens[3])
			folders[index].append(folder)

	assert len(folders) == len(iters)
	returns = defaultdict(list)
	for count, i in enumerate(iters):
		for folder in folders[i]:
			metric_file = os.path.join(folder, 'metrics.json')
			with open(metric_file, 'r') as f:
				logs = json.load(f)
				mean = logs['Eval_AverageReturn']
				std = logs['Eval_StdReturn']
				returns[count].append(mean)
	means = [None] * len(returns)
	stds = [None] * len(returns)
	for i, r in returns.items():
		means[i] = np.mean(r)
		stds = np.std(r)

	plot_mean_std(means, stds, x=iters, label='Ant-v2')
	plt.savefig('p_1_2.png')
	plt.show()

def run_2_2():
	cmd_bc = 'python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name p_2_2_bc --n_iter 1 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --eval_batch_size 5000 --num_agent_train_steps_per_iter 50000 --video_log_freq -1'
	subprocess.run(shlex.split(cmd_bc))

	cmd_dagger = 'python cs285/scripts/run_hw1_behavior_cloning.py --expert_policy_file cs285/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name p_2_2_dagger --do_dagger --n_iter 100 --expert_data cs285/expert_data/expert_data_Humanoid-v2.pkl --eval_batch_size 5000 --num_agent_train_steps_per_iter 500 --video_log_freq -1'
	subprocess.run(shlex.split(cmd_dagger))

def vis_2_2():
	bc_folder = glob.glob('cs285/data/bc_p_2_2*')
	dagger_folder = glob.glob('cs285/data/dagger_p_2_2*')
	assert len(bc_folder) == 1 and len(dagger_folder) == 1
	bc_folder = bc_folder[0]
	dagger_folder = dagger_folder[0]
	bc_metric = os.path.join(bc_folder, 'metrics.json')
	dagger_metric = os.path.join(dagger_folder, 'metrics_{itr}.json')

	with open(bc_metric, 'r') as f:
		logs = json.load(f)
		mean_bc = logs['Eval_AverageReturn']
		std_bc = logs['Eval_StdReturn']

	mean_dagger = []
	std_dagger = []
	for i in range(100):
		with open(dagger_metric.format(itr=i), 'r') as f:
			logs = json.load(f)
			mean = logs['Eval_AverageReturn']
			std = logs['Eval_StdReturn']
			mean_dagger.append(mean)
			std_dagger.append(std)

	mean_bc = [mean_bc] * len(mean_dagger)
	std_bc = [std_bc] * len(mean_dagger)
	plot_mean_std(mean_bc, std_bc, label='BC-Humanoid-v2', c='g')
	plot_mean_std(mean_dagger, std_dagger, label='Dagger-Humanoid-v2', c='y')
	plt.savefig('p_2_2.png')
	plt.show()


def plot_mean_std(means, stds, x=None, label=None, c='y'):
	sns.set(style="darkgrid")
	# fig, ax = plt.subplots()
	means = np.array(means)
	stds = np.array(stds)
	x = np.array(x) if x is not None else x


	if x is not None:
		plt.plot(x, means, label=label, c=c)
		plt.fill_between(x, means-stds, means+stds, alpha=0.3, facecolor=c)
	else:
		plt.plot(means, label=label, c=c)
		plt.fill_between(np.arange(len(means)), means-stds, means+stds, alpha=0.3, facecolor=c)

	plt.legend()
	# plt.show()



exp = {
	'run-1-2': run_1_2,
	'run-1-3': run_1_3,
	'vis-1-3': vis_1_3,
	'run-2-2': run_2_2,
	'vis-2-2': vis_2_2,
}
assert args.exp in exp
exp[args.exp]()
