# Notes

## Model Predictive Control

Important hyperparamters:

* `n_iter`: number of iterations. For each iteration, we collect samples, and train the model. Roughly 10-20
* `batch_size_initial`: number of transitions to collect for the first iteration. For this iteration, we perform random actions. Roughly 5000
* `batch_size`: number of transitions to collect for each iteration. Roughly 800
* `train_batch_size`: actual batch size. Note for each ensemble we wll sample a batch, so the total number sampled is `train_batch_size * emsemble_size` . Roughly 512. Not sure why so big
* `emsemble_size`: number of ensemble models. Roughly 3 or 5
* `horizon`: number of steps to look ahead. Roughly 10
* `num_seq`: number of action sequences to try. Roughly 1000
* `num_agent_train_steps_per_iter`: how many gradient steps to take for each iteration. Roughly 20 to 1000, depending on problem size.


The algorithm:

```python
models = [model] * ensemble_size
mpc_policy = MPC(models)
for iter in range(n_iters):
    if iter == 0:
        # Collect 5000 using random policy
        transitions = collect(env, initial_batch_size, random_policy)
    else:
        # Collect 800 using MPC policy
        transitions = collect(env, batch_sizempc_policy)
    buffer.add(transition)

    # Train
    for step in range(train_steps):
        for model in models:
            batch = buffer.sample(train_batch_size)
            # We normalize both input and output
            model.update(batch)

class MPC:
    def get_action(observation):
        candidate = uniform(numseqs, horizon, ac_dim)
        rewards = array(ensemble_size, num_seqs)
        for i, model in enumerate(models):
            for j, ac_seq in enumerate(candidate):
                # Starting from observation, do horizon steps, and accumulate the reward
                rewards[i][j] = model.apply(, observation, ac_seq, horizon)
        # Average over ensemble
        rewards = mean(rewards, axis=0)
        best_seq = candidate(argmax(rewards))
        # Only take the first action
        action = best_seq[0]
        return action
```


# Commands

## Problem 1

```
python cs285/experiments/run.py run-p1
```

Or

```
python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1

python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n5_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 5 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1

python cs285/scripts/run_hw4_mb.py --exp_name cheetah_n500_arch2x250 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 1 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 2 --size 250 --scalar_log_freq -1 --video_log_freq -1
```

## Problem 2


```
python cs285/experiments/run.py run-p2
python cs285/experiments/run.py vis-p2
```

Or

```
python cs285/scripts/run_hw4_mb.py --exp_name obstacles_singleiteration --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --n_iter 1 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10
```

## Problem 3

```
python cs285/experiments/run.py run-p3
python cs285/experiments/run.py vis-p3
```

```
python cs285/scripts/run_hw4_mb.py --exp_name obstacles --env_name obstacles-cs285-v0 --add_sl_noise --num_agent_train_steps_per_iter 20 --batch_size_initial 5000 --batch_size 1000 --mpc_horizon 10 --n_iter 12

python cs285/scripts/run_hw4_mb.py --exp_name reacher --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size_initial 5000 --batch_size 5000 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name cheetah --env_name cheetah-cs285-v0 --mpc_horizon 15 --add_sl_noise --num_agent_train_steps_per_iter 1500 --batch_size_initial 5000 --batch_size 5000 --n_iter 20
```


## Problem 4

```
python cs285/experiments/run.py run-p3
python cs285/experiments/run.py vis-p4
```

Or

```
python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon5 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 5 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon15 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 15 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_horizon30 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 30 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq100 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 100 

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_numseq1000 --env_name reacher-cs285-v0 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15 --mpc_num_action_sequences 1000

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble1 --env_name reacher-cs285-v0 --ensemble_size 1 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble3 --env_name reacher-cs285-v0 --ensemble_size 3 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15

python cs285/scripts/run_hw4_mb.py --exp_name q5_reacher_ensemble5 --env_name reacher-cs285-v0 --ensemble_size 5 --add_sl_noise --mpc_horizon 10 --num_agent_train_steps_per_iter 1000 --batch_size 800 --n_iter 15
```