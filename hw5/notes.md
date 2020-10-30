# HW5

Run command in `run_all.sh`.

Crucial hyperparamters:
* `batch_size`: number of transitions to collect for each iteration
* `n_iter`: number of iteration
* `bonus_coeff`: weight of the bonus. Crucial.
* `sigma`: for RBF kernal
* `density_train_iters`: for RBF kernal
* `density_batch_size`: for density model

The central idea is, after each iteration, you collected many samples. Then
* You update the density model with the samples
* You modify the reward function (add a bonus) of the samples, based on density