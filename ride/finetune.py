import pickle
import re
from argparse import ArgumentError
from functools import reduce
from operator import attrgetter
from pathlib import Path

import pytorch_lightning as pl
import torch
from supers import supers

from ride.core import RideMixin, Configs
from ride.unfreeze import UnfreezeMixin
from ride.utils.utils import attributedict
from ride.utils.logging import getLogger

logger = getLogger(__name__)


class FinetuneMixin(UnfreezeMixin, RideMixin):
    """Adds finetune capabilities to model

    Usage notes:
        - Inherit from the module as follows:
            `class YourModule(FinetuneMixin, RideModule, pl.LightningModule)`
        - `FinetuneMixin.__init__(self, hparams))` must be called at the end of `YourModule.__init__`
    """

    hparams: ...

    @staticmethod
    def configs() -> Configs:
        c: Configs = sum(supers(FinetuneMixin).configs())  # type: ignore
        c.add(
            name="finetune_from_weights",
            default="",
            type=str,
            description=(
                "Path to weights to finetune from. "
                "Allowed extension include {'.ckpt', '.pyth', '.pth', '.pkl', '.pickle'}."
                "For '.ckpt', the file is loaded using pytorch_lightning `load_from_checkpoint`. "
                "For '.pyth' and '.pth', the file loaded using `torch.load`. "
                "For '.pkl' and '.pickle', the file is loaded using `pickle.load`. "
            ),
        )
        c.add(
            name="finetune_from_weights_key",
            default="",
            type=str,
            description="Key in weights-file, which should contains model state_dict in case of '.pyth' or '.pth' files.",
        )
        c.add(
            name="finetune_from_weights_caffe2",
            default=0,
            type=int,
            description="Load weights from '.pkl' and '.pickle' files assuming a caffe2 format.",
        )
        c.add(
            name="finetune_params_skip",
            default="",
            type=str,
            description=(
                "Regex for matching parameter names between fintune source and target model."
                "The parameter is not copied if `finetune_params_skip` is in paramer name."
            ),
        )
        return c

    def validate_attributes(self):
        for hparam in FinetuneMixin.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    def map_loaded_weights(self, file, loaded_state_dict):
        return loaded_state_dict

    def __init__(self, hparams=None, *args, **kwargs):
        if hparams is None:
            return
        if not self.hparams:
            self.hparams = attributedict(hparams)
        self.hparams.finetune_params_skip = (
            f".*({self.hparams.finetune_params_skip}).*"
            if self.hparams.finetune_params_skip
            else None
        )

        if not self.hparams.finetune_from_weights:
            UnfreezeMixin.__init__(self, self.hparams)
            return

        # Load model
        new_model_state = self.state_dict()  # type: ignore

        # Load hparams
        default_ft_hparams = vars(FinetuneMixin.configs().default_values())
        hparams_passed = attributedict(
            {
                k: (default_ft_hparams[k] if k in default_ft_hparams else v)
                for k, v in self.hparams.items()
            }
        )

        state_dict = load_model_weights(
            self.hparams.finetune_from_weights,
            hparams_passed,
            self.hparams.finetune_from_weights_key,
        )
        state_dict = self.map_loaded_weights(
            self.hparams.finetune_from_weights, state_dict
        )

        def key_ok(k):
            return not (
                self.hparams.finetune_params_skip
                and re.match(self.hparams.finetune_params_skip, k)
            )

        def size_ok(k):
            return new_model_state[k].size() == state_dict[k].size()

        to_load = {k: v for k, v in state_dict.items() if key_ok(k) and size_ok(k)}

        self.load_state_dict(to_load, strict=False)  # type: ignore

        # Unfreeze skipped params
        names_not_loaded = set(new_model_state.keys()) - set(to_load.keys())
        names_not_loaded = {
            n for n in names_not_loaded if "num_batches_tracked" not in n
        }
        for n in {n for n in names_not_loaded if "weight" in n or "bias" in n}:
            p = rgetattr(self, n)
            p.requires_grad = True

        msg = "Copying and freezing parameters"
        if names_not_loaded:
            msg += f" (skipped {names_not_loaded})"
        logger.debug(msg)

        UnfreezeMixin.__init__(self, self.hparams)


def load_model_weights(file: str, hparams_passed, model_state_key):
    path = Path(file)
    suffix = path.suffix
    assert (
        path.exists()
    ), f"Unable to load model weights from non-existing file ({file})"
    logger.info(f"Loading model weights from {path}")

    if suffix in {".ckpt"}:
        return pl.LightningModule.load_from_checkpoint(
            file,
            hparams=hparams_passed,
        ).state_dict()
    elif suffix in {".pyth", ".pth"}:
        return try_pyth_load(file, model_state_key)
    elif suffix in {".pkl", ".pickle"}:
        return try_pickle_load(file)
    else:
        raise ArgumentError(
            f"Unable to load model weights with suffix '{suffix}'. Suffix must be one of {'.ckpt', '.pyth', '.pth', '.pkl', '.pickle'}"
        )


def try_pyth_load(file, model_state_key):
    loaded_model_state = torch.load(file, map_location="cpu")
    guesses = [
        model_state_key,
        "state_dict",
        "model_state",
    ]
    for g in guesses:
        if g in loaded_model_state.keys():
            state_dict = loaded_model_state[g]
            break

    if len(loaded_model_state) > 13:  # Hail mary
        state_dict = loaded_model_state

    if not state_dict:
        raise KeyError(
            f"None of the tried keys {guesses} fits loaded model state {loaded_model_state.keys()}. You can try another key using the `finetune_from_weights_key` hparam."
        )

    return state_dict


def try_pickle_load(file):
    with open(file, "r") as f:
        try:
            return pickle.load(file)
        except Exception:
            pass

        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            pass

    with open(file, "rb") as f:
        try:
            return pickle.load(file)
        except Exception:
            pass

        try:
            return pickle.load(f, encoding="latin1")
        except Exception:
            pass

    raise ValueError(f"Unable to load file {file}")


def rgetattr(obj, attr: str, *args):
    """Recursive getattr

    Args:
        obj (Any): Object whose attribute to select
        attr (str): String with relative path to object attribute
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))
