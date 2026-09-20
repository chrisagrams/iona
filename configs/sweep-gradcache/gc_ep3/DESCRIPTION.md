GradCache at batch 64 (60 negatives per anchor, against 3 before), 3 epochs = 536
optimizer steps, warmup 26. The epoch count is re-measured rather than carried over:
three epochs won at batch 4 because that task offered four negatives and was solved in
8,589 steps, but at batch 64 three epochs is only 536 steps and the task is far
harder, so the previous optimum says nothing here. Judged by the downstream reranker
contribution, not by the separation ratio -- the last encoder raised that ratio 5.6x and
contributed nothing to hit@1.
