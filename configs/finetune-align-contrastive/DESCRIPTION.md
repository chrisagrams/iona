The alignment tower against the CONTRASTIVE encoder rather than the pretrained one. This
is the test the whole reranking detour exists for. Alignment against the pretrained
encoder reached hit@1 0.0249 on 94 candidates, barely twice chance, because that encoder
separates peptides at an out/in ratio of only 1.34. The contrastive fine-tune raised the
ratio to 7.49. If cross-modal retrieval does not improve here, then the ratio is not the
quantity that governs it and the reranking design needs rethinking rather than tuning.
