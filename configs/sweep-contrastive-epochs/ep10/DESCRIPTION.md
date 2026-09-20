Epochs ladder on the winning contrastive configuration (lr 5e-4, kl_weight 10,
temperature 0.2), which took the pretrained encoder's out/in separation ratio from 1.34
to 7.49 in three epochs and nearly doubled downstream cross-modal hit@5. This arm runs
10 epochs. The question is whether the ratio keeps climbing or plateaus: three epochs
already drove the contrastive training loss to ~0.005 against a chance value of 1.099,
which usually means the training task is solved and further steps buy nothing -- but the
tail is untouched (clean 0.010, worst_in 0.84), so there may be room the averages hide.
