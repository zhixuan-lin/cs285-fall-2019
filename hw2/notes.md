# Notes

Important hyperparameter:

* `n_iter`: number of iterations. At each iteration we collect new trajectories. Set to 200
* `train_step`: number of gradient steps per iteration. Set to 1. **This cannot be larger then 1 since it has to be on-policy**. Also, we are using very large batch size.
* `batch_size`: number of transitions to collect per iteration. Set to 1000
* `train_batch_size`: batch size for gradient update. For PG, this is equal to batch size. So this is very large. So even though we are doing only one update, it will be good.

```python
policy = random_policy(lr=1e-3)
replay_buffer = empty_buffer(buffer_size=1000000)
agent = Agent(policy, replay_buffer)

for i in range(n_iter):
    paths = sample_trajectories(agent, expert, batch_size)
    # Concat all paths so we get a large list of 5-tuples. Add these tuples to replay buffer
    agent.add_replay_buffer(flatten_to_tuples(paths))

    for j in range(train_steps):
        # Sample a batch from replay buffer
        # Very large
        batch = agent.sample_recent_data(train_batch_size)
        # One gradient step
        q_values = sum_rewards(batch)
        nn_baselines = baseline(observations)
        advantages = q_value - nn_baselines
        # Normalize. Note this is not average baseline. It is something else.
        advantages = normalize_over_batch(advantages)
        # Do one gradient update
        agent.train(advantages, batch)
        # Do one gradient update. Yes, only one, because the batch size is huge. It doesn't work very well though. Not very useful
        agent.update_baseline(q_values, batch)
```

Special note: here, we use the baseline to predict the (normalized over batch) Q values. So, when using it, you need to denomarlize it like this:

```
advantage = q - baselines * std(q) + mean(q)
```
