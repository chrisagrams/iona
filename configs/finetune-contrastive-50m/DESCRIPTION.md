Fine-tunes the 50m spectrum encoder so its embedding space separates peptides, which the
pretrained checkpoint does not: on the replicate corpus only 1% of peptide+charge groups
have every replicate closer to each other than to any other peptide, and the in-group vs
out-group margin is 0.026 on a scale spanning [0,4]. That is why the alignment tower
reached only 2.3x chance at hit@1 -- the student was faithfully imitating a target space
with almost no peptide-discriminative structure. Supervised contrastive loss over
replicates supplies the missing property directly; KL to the frozen pretrained intensity
head stops the encoder buying separation by discarding the peak chemistry that makes the
checkpoint worth starting from. The output is a drop-in --pretrained_path for the
alignment run.
