The 100m denoise fine-tune under DeepSpeed ZeRO-2, and the template for the 100m grid.
Identical to the 50m DeepSpeed config except for the checkpoint, so any difference between
the two size ladders is attributable to encoder capacity alone. The 50m grid's 216 arms
decided which axes this grid still needs to sweep.
