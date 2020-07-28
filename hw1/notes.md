
# Training loop

Critical hyperparameters:

* `n_iter`: number of Dagger iteration loops. Set to 100
* `train_steps`: how many gradient steps to take per iteration. Set to 1000. (The larger the better)
* `batch_size`: how many new transitions to collect for each iteration. Set to 1000.
* `train_batch_size`: training batch size, set to 32

```python
policy = random_policy(lr=1e-3)
replay_buffer = empty_buffer(buffer_size=1000000)
agent = Agent(policy, replay_buffer)
for i in range(n_iter):
  if i == 0:
  # A list of lists. Inner lists contains 5-tuples.
    # batch_size is not used here. the more the better
    paths = sample_trajectories(expert, expert, batch_size)
  else:
  # A list of lists. Inner lists contains 5-tuples. The total length will be largers than B
    paths = sample_trajectories(agent, expert, batch_size)
  # Concat all paths so we get a large list of 5-tuples. Add these tuples to replay buffer
  agent.add_replay_buffer(flatten_to_tuples(paths))
  for j in range(train_steps):
    # Sample a batch from replay buffer
    batch = agent.sample(train_batch_size)
    # One gradient step
    agent.train(batch)
```

Control:

* Number of data: fixed, 2000
* Batch size: 100
* Number of iterations: 1000
* Network depth: 2
* Width: 64
* Eval batch size: the total length of episode used for evaluation
* Tasks
  * Ant
  * Hopper
  * Walker
  * HalfCheetah
  * Humanoid
