"""Embed exported MGF spectra with pretrained GLEAMS (CPU), aligned to meta.parquet rows.

    /lus/flare/projects/UIC-HPC/khuss/msdelta/baselines/gleams/env/bin/python \
        embed_gleams.py DATA_DIR [N_JOBS]

Follows `gleams embed` (gleams/gleams.py + nn.embed) exactly -- same config, same seeded
reference-spectrum selection (gleams.gleams import calls rndm.set_seeds() before the
ReferenceSpectraEncoder shuffles), same encoders and network -- except:
  * no "No GPU found" abort (CPU TensorFlow);
  * no charge 2..5 filter and no drop of spectra GLEAMS' preprocessing marks invalid
    (<10 peaks or <250 m/z range): those are encoded after a lenient preprocessing
    (min_peaks=1, min_mz_range=0) so every row gets a vector; `strict_valid.npy` flags
    which rows GLEAMS itself would have embedded.
Writes gleams_embed.npy (n_rows x 32, NaN for rows without an MGF entry) and
strict_valid.npy.
"""
import os
import sys
import time

import numpy as np

import gleams.gleams  # noqa: F401  -- logging init + rndm.set_seeds(), as the CLI does
from gleams import config
from gleams.feature import encoder, spectrum
from gleams.ms_io import mgf_io, ms_io
from gleams.nn import data_generator, embedder
import joblib
import pyarrow.parquet as pq
import scipy.sparse as ss

PREP = {'mz_min': config.fragment_mz_min, 'mz_max': config.fragment_mz_max,
        'min_peaks': config.min_peaks, 'min_mz_range': config.min_mz_range,
        'remove_precursor_tolerance': config.remove_precursor_tolerance,
        'min_intensity': config.min_intensity,
        'max_peaks_used': config.max_peaks_used, 'scaling': config.scaling}
LENIENT = dict(PREP, min_peaks=1, min_mz_range=0.0)


def build_encoder():
    return encoder.MultipleEncoder([
        encoder.PrecursorEncoder(
            num_bits_mz=config.num_bits_precursor_mz, mz_min=config.precursor_mz_min,
            mz_max=config.precursor_mz_max, num_bits_mass=config.num_bits_precursor_mass,
            mass_min=config.precursor_mass_min, mass_max=config.precursor_mass_max,
            charge_max=config.precursor_charge_max),
        encoder.FragmentEncoder(min_mz=config.fragment_mz_min,
                                max_mz=config.fragment_mz_max, bin_size=config.bin_size),
        encoder.ReferenceSpectraEncoder(
            filename=config.ref_spectra_filename, preprocessing=PREP,
            fragment_mz_tol=config.fragment_mz_tol,
            num_ref_spectra=config.num_ref_spectra)])


def encode_file(path, enc):
    rows, valid, encs = [], [], []
    for spec in ms_io.get_spectra(path):
        row = int(spec.identifier.split(';')[0].split('=')[1])
        raw = (spec.mz.copy(), spec.intensity.copy())
        p = spectrum.preprocess(spec, **PREP)
        ok = bool(p.is_valid)
        if not ok:
            spec2 = mgf_io.MsmsSpectrum(spec.identifier, spec.precursor_mz,
                                        spec.precursor_charge, raw[0], raw[1], None, 0.0)
            spec2.is_processed = False
            p = spectrum.preprocess(spec2, **LENIENT)
            if not p.is_valid:        # nothing left at all (e.g. every peak < 50.5 m/z)
                continue
        rows.append(row)
        valid.append(ok)
        encs.append(enc.encode(p))
    return rows, valid, encs


def main(data_dir, n_jobs):
    t0 = time.time()
    n = pq.read_table(os.path.join(data_dir, 'meta.parquet'), columns=['row']).num_rows
    enc = build_encoder()
    print(f'[gleams] {len(enc.encoders[2].ref_spectra)} reference spectra', flush=True)
    # Split each MGF into chunks for parallel encoding.
    chunk_paths = []
    tmp = os.path.join(data_dir, 'chunks')
    os.makedirs(tmp, exist_ok=True)
    for name in ('experimental.mgf', 'consensus.mgf'):
        buf, k, count = [], 0, 0
        with open(os.path.join(data_dir, name)) as f:
            for line in f:
                buf.append(line)
                if line.startswith('END IONS'):
                    count += 1
                    if count % 500 == 0:
                        p = os.path.join(tmp, f'{name}.{k}.mgf')
                        open(p, 'w').writelines(buf)
                        chunk_paths.append(p)
                        buf, k = [], k + 1
        if buf:
            p = os.path.join(tmp, f'{name}.{k}.mgf')
            open(p, 'w').writelines(buf)
            chunk_paths.append(p)
    print(f'[gleams] {len(chunk_paths)} chunks; encoding with {n_jobs} jobs', flush=True)
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(encode_file)(p, enc) for p in chunk_paths)
    rows, valid, encs = [], [], []
    for r, v, e in results:
        rows += r
        valid += v
        encs += e
    print(f'[gleams] encoded {len(rows):,} spectra ({sum(valid):,} strict-valid) in '
          f'{time.time() - t0:.0f}s', flush=True)

    emb = embedder.Embedder(num_precursor_features=config.num_precursor_features,
                            num_fragment_features=config.num_fragment_features,
                            num_ref_spectra_features=config.num_ref_spectra,
                            lr=config.lr, filename=config.model_filename)
    emb.load()
    split = (config.num_precursor_features,
             config.num_precursor_features + config.num_fragment_features)
    seq = data_generator.EncodingsSequence(ss.vstack(encs, 'csr'), config.batch_size, split)
    out = np.vstack(emb.embed(seq)).astype(np.float32)
    full = np.full((n, out.shape[1]), np.nan, dtype=np.float32)
    full[np.asarray(rows)] = out
    strict = np.zeros(n, dtype=bool)
    strict[np.asarray(rows)] = np.asarray(valid)
    np.save(os.path.join(data_dir, 'gleams_embed.npy'), full)
    np.save(os.path.join(data_dir, 'strict_valid.npy'), strict)
    print(f'[gleams] wrote {full.shape} embeddings; {np.isnan(full[:, 0]).sum()} rows '
          f'missing; total {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32)
