"""Tokenize modified peptides for de novo sequencing.

Each token is one residue, one modified residue, or one N-terminal modification, written in
MassIVE-KB notation (`C[+57.021]`, `[-17.027]QK`). Other notations are converted when their
dataset is loaded, not here.
"""

from __future__ import annotations

from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from iona.chemistry import RESIDUE_MASSES as AA

# Monoisotopic modification shifts from Unimod.
CARBAMIDOMETHYL, OXIDATION, DEAMIDATION = 57.021464, 15.994915, 0.984016
ACETYL, CARBAMYL, AMMONIA_LOSS = 42.010565, 43.005814, -17.026549

# Every residue and N-terminal modification token, in vocabulary order, with its mass.
TOKEN_MASSES = {
    "G": AA["G"], "A": AA["A"], "S": AA["S"], "P": AA["P"], "V": AA["V"], "T": AA["T"],
    "C": AA["C"], "C[+57.021]": AA["C"] + CARBAMIDOMETHYL, "L": AA["L"], "I": AA["I"],
    "N": AA["N"], "D": AA["D"], "Q": AA["Q"], "K": AA["K"], "E": AA["E"], "M": AA["M"],
    "H": AA["H"], "F": AA["F"], "R": AA["R"], "Y": AA["Y"], "W": AA["W"],
    "M[+15.995]": AA["M"] + OXIDATION,
    "N[+0.984]": AA["N"] + DEAMIDATION,
    "Q[+0.984]": AA["Q"] + DEAMIDATION,
    "[+42.011]": ACETYL,
    "[+43.006]": CARBAMYL,
    "[-17.027]": AMMONIA_LOSS,
    "[+43.006][-17.027]": CARBAMYL + AMMONIA_LOSS,
}

# One residue with an optional modification, or an N-terminal modification of up to two
# stacked shifts, as MassIVE-KB writes carbamylation with ammonia loss.
TOKEN_PATTERN = r"[A-Z](?:\[[+-]\d+\.\d+\])?|(?:\[[+-]\d+\.\d+\]){1,2}"


class IonaPeptideTokenizer(TokenizersBackend):
    """Word-level peptide tokenizer that appends EOS."""

    model_input_names = ["input_ids", "attention_mask"]
    model = WordLevel

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        **kwargs,
    ):
        # Special tokens arrive as AddedToken when loading from a checkpoint.
        pad, eos, unk = str(pad_token), str(eos_token), str(unk_token)
        if vocab is None:
            tokens = (pad, eos, unk, *TOKEN_MASSES)
            vocab = {token: index for index, token in enumerate(tokens)}
        self._tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=unk))
        self._tokenizer.pre_tokenizer = pre_tokenizers.Split(
            Regex(TOKEN_PATTERN), behavior="isolated"
        )
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A {eos}", special_tokens=[(eos, vocab[eos])]
        )
        self._tokenizer.decoder = decoders.Fuse()
        super().__init__(pad_token=pad_token, eos_token=eos_token, unk_token=unk_token, **kwargs)

    @property
    def token_masses(self) -> list[float]:
        """Mass of every token id; special tokens weigh nothing."""
        return [
            TOKEN_MASSES.get(token, 0.0)
            for token in self.convert_ids_to_tokens(list(range(len(self))))
        ]

    @property
    def n_terminal_token_ids(self) -> list[int]:
        """Ids of tokens that are only valid as the first token of a peptide."""
        return sorted(index for token, index in self.get_vocab().items() if token.startswith("["))


IonaPeptideTokenizer.register_for_auto_class("AutoTokenizer")
