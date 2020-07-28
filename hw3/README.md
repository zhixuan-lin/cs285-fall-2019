# HW 3

## Question 3

```python
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q3_hparam1_update3000 --target_update_freq 3000
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q3_hparam2_update1000 --target_update_freq 1000
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q3_hparam3_update5000 --target_update_freq 5000
python cs285/scripts/run_hw3_dqn.py --env_name LunarLander-v2 --exp_name q3_hparam4_update2000 --target_update_freq 2000
```

## Question 4

```python
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 1_1 -ntu 1 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 1_100 -ntu 1 -ngsptu 100
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 100_1 -ntu 100 -ngsptu 1
python cs285/scripts/run_hw3_actor_critic.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name 10_10 -ntu 10 -ngsptu 10
```

## Question 5

```python
python cs285/scripts/run_hw3_actor_critic.py --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10
python cs285/scripts/run_hw3_actor_critic.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10
```
