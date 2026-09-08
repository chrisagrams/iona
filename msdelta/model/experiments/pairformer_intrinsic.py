"""Spectrum-intrinsic Pairformer: charge-aware pair features, no identification leakage.

This is a controlled variant of ``pairformer.py`` that changes exactly two things and reuses
everything else (blocks, single/pair refinement, the AF3 build-phase ladder, the diagnostics
surface, the pretraining objective):

1. **Charge-aware pair features (your request).** The dictionary soft-match (p3) and the
   isotope-spacing feature (p5) no longer assume singly-charged fragments. A neutral mass ``m``
   between two charge-``z`` fragments appears as a gap of ``m / z`` Th, and a ``z+`` ion's
   isotope satellites are spaced ``1.00336 / z`` Th. Both features now test a configurable set
   of charge hypotheses ``z in {1 .. pair_charge_states}``:
     * p3 takes the MAX over charge hypotheses -> stays ``F_loss`` wide (memory-neutral), so it
       fires when a gap matches a residue/loss at ANY plausible charge (charge-robust ladders);
     * p5 CONCATENATES over charge -> ``2 * pair_charge_states`` wide, so the network can read
       the fragment charge straight off which isotope hypothesis lights up.
   The charge hypotheses are fixed constants, not inputs -- nothing identification-derived is
   needed to evaluate them.

2. **No identification-derived precursor (your leakage concern).** In THIS dataset
   ``precursor_mz`` and ``charge`` are reconstructed from the ``peptide_charge`` label
   (``msdelta/data/loading.py``), i.e. from the answer -- noise-free and always consistent with
   the true peptide. A model meant to *recognise* the molecule from its spectrum must not lean
   on that. So this variant, by default:
     * DROPS the p4 complementarity feature (it needs the neutral precursor mass), and
     * builds NO global conditioning vector g (which embedded precursor m/z + charge).
   The single and pair reps still carry BOTH m/z and intensity -- the change is only that the
   *precursor* is no longer fed in. The model is expected to infer charge from the spectrum via
   the multi-charge hypotheses above.

   ``use_precursor_cond`` (default False) re-enables the precursor global-conditioning arm for a
   controlled ablation. It is deliberately leaky in this dataset -- turn it on only to *measure*
   how much the precursor helps, never as the default recognition model. Even when on, the p4
   complementarity feature is NOT restored here; use ``pairformer.py`` for that full arm.

Note on realism: a *measured* precursor m/z (from the MS1 scan) is a legitimate observable and
would not be leakage -- the problem is specific to this dataset reconstructing it from the label.
If a measured precursor column becomes available, feeding it through ``use_precursor_cond`` would
be sound; until then, keep the default.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from msdelta.model.experiments.pairformer import (
    _C13_SPACING,
    _NEUTRAL_LOSS_BANK,
    GlobalCond,
    MSDeltaPairformerConfig,
    MSDeltaPairformerForPreTraining,
    MSDeltaPairformerModel,
    PairStack,
    PeakEmbedMZ,
)
from msdelta.model.fourier import FourierFeatures
from msdelta.model.modeling import (
    IntensityHead,
    MSDeltaForPreTrainingOutput,
    MSDeltaPreTrainedModel,
)


class MSDeltaPairformerIntrinsicConfig(MSDeltaPairformerConfig):
    """Pairformer config plus the spectrum-intrinsic (charge-aware, no-precursor) settings."""

    model_type = "msdelta-pairformer-intrinsic"

    def __init__(
        self,
        pair_charge_states: int = 3,
        use_precursor_cond: bool = False,
        **kwargs,
    ):
        # Set before super().__init__, which calls _validate().
        self.pair_charge_states = pair_charge_states
        self.use_precursor_cond = use_precursor_cond
        super().__init__(**kwargs)

    def _validate(self) -> None:
        super()._validate()
        if self.pair_charge_states < 1:
            raise ValueError("pair_charge_states must be >= 1")


class PairFeaturesIntrinsic(nn.Module):
    """Charge-aware, precursor-free per-pair features ``pair_feats[i, j]``.

    Blocks (order matters -- p1 stays first so ``PairStack.evaluate`` can slice the signed-Δ
    component for the diagnostics):
      p1 signed Δ Fourier | p2 mass-defect Fourier | p3 charge-aware dictionary soft-match
      | p5 charge-aware isotope spacing | p6 log intensity ratio.
    There is no p4 (complementarity) -- it requires the neutral precursor mass, which this
    variant deliberately does not use.
    """

    def __init__(self, config: MSDeltaPairformerIntrinsicConfig):
        super().__init__()
        self.use_intensity = config.pair_use_intensity
        self.sigma_ppm = config.loss_bank_sigma_ppm
        self.charge_states = config.pair_charge_states
        # Signed Δ bank -- exposed as ``.ff`` for the diagnostics, so keep it first.
        self.ff = FourierFeatures(
            config.delta_bias_n_freqs,
            config.delta_bias_f_min,
            config.delta_bias_f_max,
            log_spaced=True,
            learnable=config.delta_bias_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        self.ff_defect = FourierFeatures(
            config.mass_defect_n_freqs,
            1.0,
            float(config.mass_defect_n_freqs),
            log_spaced=True,
            learnable=config.delta_bias_learnable,
            log_parameterized=config.fourier_log_parameterized,
        )
        self.register_buffer(
            "loss_bank", torch.tensor(_NEUTRAL_LOSS_BANK, dtype=torch.float32), persistent=True
        )
        # Charge hypotheses z in {1 .. charge_states}; a gap of mass m at charge z is m/z Th.
        self.register_buffer(
            "charge_z",
            torch.arange(1, self.charge_states + 1, dtype=torch.float32),
            persistent=False,
        )
        self.out_dim = (
            self.ff.out_dim
            + self.ff_defect.out_dim
            + self.loss_bank.numel()          # p3: max over charge -> stays F_loss wide
            + 2 * self.charge_states          # p5: isotope k in {1, 2} x each charge
            + (1 if self.use_intensity else 0)
        )

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        precursor_mz: Tensor | None = None,  # accepted for call-compatibility; unused
        charge: Tensor | None = None,        # (this variant does not consume the precursor)
    ) -> Tensor:
        delta = mz.unsqueeze(-1) - mz.unsqueeze(-2)  # (B, N, N), signed
        abs_delta = delta.abs()
        feats = [self.ff(delta), self.ff_defect(abs_delta - abs_delta.floor())]

        # sigma in Da, m/z-dependent: the tolerance grows with the heavier peak's mass.
        heavier = torch.maximum(mz.unsqueeze(-1), mz.unsqueeze(-2)).clamp_min(1.0)
        sigma = (self.sigma_ppm * 1e-6) * heavier
        two_var = 2.0 * (sigma * sigma).clamp_min(1e-12)  # (B, N, N)

        bank = self.loss_bank.to(delta.dtype)             # (F,)
        zc = self.charge_z.to(delta.dtype)                # (Z,)

        # p3 charge-aware dictionary soft-match. Loop over charge with a running max so peak
        # memory stays (B, N, N, F) rather than materialising (B, N, N, F, Z).
        p3 = None
        for zi in range(self.charge_states):
            target = bank / zc[zi]                        # (F,) expected |Δ| at this charge
            diff = abs_delta.unsqueeze(-1) - target       # (B, N, N, F)
            g = torch.exp(-(diff * diff) / two_var.unsqueeze(-1))
            p3 = g if p3 is None else torch.maximum(p3, g)
        feats.append(p3)

        # p5 charge-aware isotope spacing 1.00336*k / z, concatenated over charge so the model
        # can read the fragment charge from which hypothesis fires.
        iso = []
        for k in (1, 2):
            centers = (_C13_SPACING * k) / zc             # (Z,)
            d = abs_delta.unsqueeze(-1) - centers         # (B, N, N, Z)
            iso.append(torch.exp(-(d * d) / two_var.unsqueeze(-1)))
        feats.append(torch.cat(iso, dim=-1))              # (B, N, N, 2Z)

        # p6 relative intensity, log(I_i / I_j). log_intensity is already log-scale.
        if self.use_intensity:
            rel = log_intensity.unsqueeze(-1) - log_intensity.unsqueeze(-2)
            feats.append(rel.unsqueeze(-1))

        return torch.cat([f.to(feats[0].dtype) for f in feats], dim=-1)


class PairStackIntrinsic(PairStack):
    """``PairStack`` with the intrinsic (charge-aware, precursor-free) pair featurizer."""

    def __init__(self, config: MSDeltaPairformerIntrinsicConfig):
        super().__init__(config)
        # Swap the featurizer and resize the W_c projection to its output width. The base
        # ``pair_feats``/``w_c`` built by super().__init__ are discarded here; post_init (run by
        # the owning model AFTER this stack is built) initialises the new w_c.
        self.pair_feats = PairFeaturesIntrinsic(config)
        self.w_c = nn.Linear(self.pair_feats.out_dim, config.pair_channels, bias=False)


class MSDeltaPairformerIntrinsicModel(MSDeltaPairformerModel):
    """Encoder with charge-aware pair features and no identification-derived conditioning."""

    config_class = MSDeltaPairformerIntrinsicConfig

    def __init__(self, config: MSDeltaPairformerIntrinsicConfig):
        # Mirror the parent __init__ but swap the pair stack and gate global conditioning on the
        # explicit (leaky) opt-in. Build everything BEFORE post_init so weights initialise.
        MSDeltaPreTrainedModel.__init__(self, config)
        self.embed = PeakEmbedMZ(config)
        self.bias_module = PairStackIntrinsic(config)
        self.global_cond = GlobalCond(config) if config.use_precursor_cond else None
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()
        self._zero_init_residual_readouts()


class MSDeltaPairformerIntrinsicForPreTraining(MSDeltaPairformerForPreTraining):
    """Masked-intensity pretraining on the spectrum-intrinsic encoder.

    Reuses the baseline objective. By default it feeds NO precursor/charge to the encoder, so
    the shared collator's ``precursor_mz``/``charge`` keys are accepted and ignored -- there is
    no identification leakage. With ``use_precursor_cond=True`` it threads them into the global
    conditioning arm for an ablation (leaky in this dataset; see the module docstring).
    """

    config_class = MSDeltaPairformerIntrinsicConfig

    def __init__(self, config: MSDeltaPairformerIntrinsicConfig):
        MSDeltaPreTrainedModel.__init__(self, config)
        self.msdelta = MSDeltaPairformerIntrinsicModel(config)
        self.intensity_head = IntensityHead(config.hidden_size)
        self.post_init()
        self.msdelta._zero_init_residual_readouts()

    def forward(
        self,
        mz: Tensor,
        log_intensity: Tensor,
        attention_mask: Tensor | None = None,
        mask_positions: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
        **conditioning: Tensor,
    ):
        if return_dict is None:
            return_dict = self.config.return_dict
        # Only pass the precursor through when explicitly opted in; otherwise it is dropped here
        # so nothing identification-derived reaches the encoder.
        if self.config.use_precursor_cond:
            precursor_mz = conditioning.get("precursor_mz")
            charge = conditioning.get("charge")
        else:
            precursor_mz = None
            charge = None
        outputs = self.msdelta(
            mz=mz,
            log_intensity=log_intensity,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
            precursor_mz=precursor_mz,
            charge=charge,
            return_dict=True,
        )
        logits = self.intensity_head(outputs.last_hidden_state)
        loss = self._masked_intensity_loss(logits, labels, mask_positions)
        if not return_dict:
            result = (logits,)
            return ((loss,) + result) if loss is not None else result
        return MSDeltaForPreTrainingOutput(loss=loss, logits=logits)
