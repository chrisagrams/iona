The winning contrastive configuration with GradCache, at a batch of 64 instead of 4.
The epochs ladder established that steps are not the constraint -- three epochs reached a
separation ratio of 6.94 and ten epochs fell to 4.82, because the contrastive loss was
already 0.005 against a chance value of 1.099 and further optimisation only overfits a
task made trivial by having four negatives. GradCache embeds the batch in chunks of 4
under no_grad, differentiates the loss with respect to the cached embeddings, then
re-embeds each chunk with grad and pushes that gradient through, so peak memory is one
chunk while the loss sees all 64. That is 63 negatives per anchor instead of 3, at full
512 peaks with no data dropped beyond the usual 19.4%. The gradient is exact to float32
accumulation order, asserted against a full-batch backward in the tests.
