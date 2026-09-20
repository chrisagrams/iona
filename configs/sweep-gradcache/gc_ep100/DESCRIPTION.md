GradCache at batch 64, 100 epochs = 17893 optimizer steps, warmup 894. Extends the
ladder past the point where the batch-4 runs peaked (8,589 steps), because batch 64
behaves differently: batch 4 drove the contrastive loss to 0.4% of chance in 8,589 steps
and then degraded with more, while batch 64 was still at 37% of chance after 1,789 and
falling. The question is whether a task that is actually hard keeps paying for steps
where the easy one stopped. Judged by the downstream reranker contribution, not by the
separation ratio.
