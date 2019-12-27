* Define a module in TF:
  
    * A function that returns placeholders
        * Input: 
        * Ground truth
    * A forward:
        * That takes the placeholder and output the prediction place holder
    * A `train_op`
        * That takes all placeholders
        * That uses forward
        * Compute loss
        * Return a optimization OP

* Replaybuffer:
    * Is a very large queue of sequential `(o, a, o_next, r, t)` 5-tuple
    * Stores sequentially the following
        * observations
        * actions
        * next observations
        * rewards
        * terminals 
    * Has a method that 
        * Takes several episodes
        * Appends the above things sequentially
    * Has a method that samples a batch of the above 

* Policy:
    * Has an MLP graph
    * Has an optimization OP
    * Given a batch of observations, return a batch of actions, using the MLP graph
    * Given a batch of observations and a batch of actions, update the policy, using the optimization OP

* Agent: 
    * Has a replay buffer
    * Has a (actor) policy
    * Has a method such that:
        * Given some episodes, update its buffer
    * Has a method such that:
        * Given a batch of 5-tuples, update its policy
    * Has a method such that:
        * Samples a batch from the replaybuffer


* Training loop:

```python
policy = random_policy()
agent = Agent()
# This refers to the number of new 5-tuples to collect
batch_size = B
# This refers to the actual batch size in training
train_batch_size = Btrain
# Training steps
train_steps = T
for i in range(n_iter):
  if i == 0:
		# A list of lists. Inner lists contains 5-tuples.
    paths = load_expert_data()
  else:
		# A list of lists. Inner lists contains 5-tuples. The total length will be largers than B
    paths = sample_trajectories(batch_size)
  # Concat all paths so we get a large list of 5-tuples. Add these tuples to replay buffer
  agent.add_replay_buffer(paths)
  for j in range(train_steps):
    # Sample a batch from replay buffer
    batch = agent.sample() 
    # One gradient step
    agent.train(batch)
```

So data:

* Data
  * initial: expert
  * Else: collect a certain number with the policy
* add to buffer
* train using the **buffer**.







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