# Notes

## DQN

Important hyperparameters:

* `learning_start`: before that, we just collect transitions using random policy, and do not learn any Q function. Set to 50000 for Pong and 1000 for LunarLander
* `target_update_freq`: when reaching this step, update target function weights
* `num_timesteps`: number of steps
* `frame_history_len`: number of frames used to contruct agent state
* `exploration schedule`. epsilon value for each step. For atar:
  * Atari: (0, 1e6, total/8), (1.0, 0.1, 0.01)
  * Lander: (0, total/10), (1.0, 0.02)
* `batch_size`: for each step. Set to 32

```python
Q_target = Q()
Q_online = Q()
replay_buffer.init()

o = env.reset()
for i in range(n_iter):
    if n_iter < learning_starts:
        action = env.action_space.sample()
    else:
        epsilon = exploration_schedule(i)
        # Note, epsilon can be one
        action = greedy(Q_online(o), epsilon)
    (o, r, d, i) = env.step(env.action_space.sample())
    buffer.add((o, r, d, i))

    # Do gradient step
    o, o_next, d, i = buffer.sample(batch_size)

    # Double Q
    if ddqn:
        selector = Q_online
    else:
        selector = Q_target

    update_target = r(o, action) + gamma * Q_target(argmax(selector))
    update_target = update_target.detach()

    # Do one gradient update. Yes, only one
    (Q_online(o) - update_target).backward.step()

    # Update target when appropriate
    if i % target_update_freq == 0:
        Q_target = Q_online.copy()
```


## Actor-Critic

Important hyperparameter:

* `n_iter`: number of iterations. At each iteration we collect new trajectories. Set to 200
* `batch_size`: number of transitions to collect per iteration. Set to 1000
* `train_batch_size`: batch size for gradient update. For PG, this is equal to batch size. So this is very large. So even though we are doing only one update, it will be good.
* `train_steps`: has to be 1 because it has to be on-policy. So I ignored this.
* `num_target_update`: how many times to update the critic target, per iteration
* `grad_steps_per_target_update`: how many gradient steps to take after we update the target

```python
policy = random_policy(lr=1e-3)
# There doesn't seem to be some special initialization
critic = random_critic()
replay_buffer = empty_buffer(buffer_size=1000000)
agent = Agent(policy, replay_buffer)

for i in range(n_iter):
    paths = sample_trajectories(agent, expert, batch_size)
    # Concat all paths so we get a large list of 5-tuples. Add these tuples to replay buffer
    agent.add_replay_buffer(flatten_to_tuples(paths))

        # Sample a batch from replay buffer
        # Very large
        batch = agent.sample_recent_data(train_batch_size)

        # Update critic
        for _ in range(num_target_updates):
            Vnext = critic(batch['o'])
            target = batch['reward'] + gamma * Vnext
            target = target.detach()
            for _ in range(num_grad_steps_per_target_update)
                (target - V(batch)).batckward.step()

        # Update actor

        advantages = reward + gamma * critic(batch['o'])
        # Advantage normalization. 
        advantages = normalize_over_batch(advantages)
        # Do one gradient update. We can only do one because this has to be on policy
        agent.train(advantages, batch)
```

Note this is different from DQN: 

* Actor-Critc: a huge batch, and we do target update and grad steps iteratively on this same huge batch. So, we only need to evaluete the target once. So we don't need to maintain two networks.
* DQN: each time, we only take a small batch. For this batch, we evaluate it using target network and update the online network. We need to do evaluation for each time step. So we have to maintain two networks.