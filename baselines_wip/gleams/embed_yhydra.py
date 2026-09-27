"""Embed exported MGF spectra with yHydra's pretrained spectrum tower (CPU), rows aligned.

    /lus/flare/projects/UIC-HPC/khuss/msdelta/baselines/yhydra/env/bin/python \
        embed_yhydra.py DATA_DIR

Preprocessing as yHydra's proteomics_utils.get_features with its default config: L2-normalise
intensities, keep the 100 most intense peaks (in m/z order), zero-pad to 500, input
(500, 2) = [m/z, intensity]. Model saved_27_06_2021 (the one load_model.py uses); the
spectrum embedder is the sub-model spectrum_input -> spec_emb. No precursor input.
"""
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf
from pyteomics import mgf

MODEL = '/lus/flare/projects/UIC-HPC/khuss/msdelta/baselines/yhydra/saved_27_06_2021'


def features(mz, it, max_peaks=100, pad=500):
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(it, dtype=np.float64)
    it = it / np.linalg.norm(it)
    idx = np.argsort(it)[-max_peaks:][::-1]
    mask = np.zeros(len(mz), bool)
    mask[idx] = True
    mz, it = mz[mask], it[mask]
    out = np.zeros((pad, 2), np.float32)
    out[:len(mz), 0] = mz
    out[:len(mz), 1] = it
    return out


def main(data_dir):
    n = pq.read_table(os.path.join(data_dir, 'meta.parquet'), columns=['row']).num_rows
    rows, x = [], []
    for name in ('experimental.mgf', 'consensus.mgf'):
        with mgf.MGF(os.path.join(data_dir, name)) as f:
            for s in f:
                rows.append(int(s['params']['title'].split(';')[0].split('=')[1]))
                x.append(features(s['m/z array'], s['intensity array']))
    x = np.stack(x)
    model = tf.keras.models.load_model(MODEL, custom_objects={'metric_acc': lambda a: a})
    spec = tf.keras.Model(inputs=model.get_layer('spectrum_input').input,
                          outputs=model.get_layer('spec_emb').output)
    print('[yhydra] input', spec.input_shape, 'output', spec.output_shape, flush=True)
    out = spec.predict(x, batch_size=256)
    full = np.full((n, out.shape[-1]), np.nan, np.float32)
    full[np.asarray(rows)] = out.reshape(len(rows), -1)
    np.save(os.path.join(data_dir, 'yhydra_embed.npy'), full)
    print(f'[yhydra] wrote {full.shape}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
