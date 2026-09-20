Contrastive fine-tune at max_peaks=256 instead of 512, to buy a bigger batch. DeltaMZBias
is O(batch x peaks^2), so quartering the peak budget affords 64 spectra per batch instead
of 16, which is 96 positive pairs against 24 and far more negatives -- and a contrastive
objective is largely a function of how many negatives each step sees. The cost is data:
the processor DROPS spectra above max_peaks rather than truncating, and the corpus has a
median of 222 peaks, so this discards a large additional fraction on top of the 19.4%
already lost at 512, biased toward highly charged and longer peptides. That makes this a
measurement of whether batch size is what the objective needs, not a candidate for the
final encoder: an encoder trained at 256 peaks also does not match the 512-peak setting
the alignment tower and inference use.
