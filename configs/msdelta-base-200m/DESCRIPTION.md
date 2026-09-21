Pretraining the 200m MSDelta encoder on chrisagrams/MSConsensus-100M with the masked-peak objective, DeepSpeed
ZeRO-2 across nodes, lr 1.3e-4, 3 epochs. This is the size ladder: the five
msdelta-base configs differ in capacity and in nothing else, so a comparison across them
attributes any difference to model size alone. Checkpoints from these runs are what every
fine-tune starts from, which is why the pretraining data and the fine-tuning data must
stay disjoint.
