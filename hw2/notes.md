Central idea: treat this as a weighted likelihood maximization problem. The (s, a) pair to
optimize comes from the policy, and the weights are computed by 
discounted reward-to-go. Since they are weights, it makes sense to normalize them.

Baseline: kind of weird. The code does baseline gradient update only once for one policy gradient update. But I don't care.
And it normalize the target. Though theoretically it won't be bad. But still weird.
