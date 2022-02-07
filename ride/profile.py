import contextlib
from abc import abstractmethod
from time import time
from typing import Any, Dict, Optional, Tuple, overload

import numpy as np
import torch
from tqdm import tqdm

from ride.utils.gpus import parse_gpus, parse_num_gpus
from ride.utils.logging import getLogger
from ride.utils.utils import name, some

logger = getLogger(__name__)


def inference_memory(
    model: torch.nn.Module, abbreviated=True
) -> Tuple[Dict[str, int], Optional[str]]:
    prevously_training = model.training
    model.eval()

    # Move model to GPU if available
    prev_device = model.device
    gpus = parse_gpus(model.hparams.gpus) if hasattr(model.hparams, "gpus") else None
    if not gpus:  # Cannot profile
        if prevously_training:
            model.train()
        return {}, None

    dev = f"cuda:{gpus[0]}"
    data = torch.randn(1, *model.input_shape, device=dev)
    model.to(device=dev)

    memory_summary = None
    stats = {}

    with torch.no_grad():
        try:
            model.warm_up(tuple(data.shape))
            torch.cuda.reset_peak_memory_stats(device=dev)
            model(data)
            memory_summary = torch.cuda.memory_summary(
                device=dev, abbreviated=abbreviated
            )
            stats = {
                "memory_allocated": torch.cuda.memory_allocated(device=dev),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device=dev),
            }
        except Exception as e:
            raise ValueError(
                f"Caught exception while attempting to profile model memory. "
                "Did you ensure that you model has a correct `input_shape` attribute (omitting the batch_size dimension)? "
                f"Exception contents: {e}"
            )
        finally:
            model.to(device=prev_device)
            if prevously_training:
                model.train()

    return stats, memory_summary


@overload
def time_single_inference(model: torch.nn.Module, detailed=False) -> float:
    ...  # pragma: no cover


@overload
def time_single_inference(
    model: torch.nn.Module, detailed=True
) -> Tuple[float, torch.autograd.profiler.EventList,]:
    ...  # pragma: no cover


def time_single_inference(model: torch.nn.Module, detailed=True, warm_up=True):
    for attr in [
        "device",
        "input_shape",
        "hparams.batch_size",
        "forward",
    ]:
        assert some(
            model, attr
        ), f"{name(model)} should define `{attr}` but none was found."

    prevously_training = model.training
    model.eval()

    # Move model to GPU if available
    prev_device = model.device
    gpus = parse_gpus(model.hparams.gpus) if hasattr(model.hparams, "gpus") else None
    new_device = f"cuda:{gpus[0]}" if gpus else "cpu"
    model.to(device=new_device)

    # Initialise data on CPU
    data = torch.randn(model.hparams.batch_size, *model.input_shape, device="cpu")
    if warm_up:
        model.warm_up(tuple(data.shape))

    try:
        with torch.no_grad(), torch.autograd.profiler.profile(
            use_cuda=(model.device.type != "cpu"), profile_memory=True
        ) if detailed else contextlib.suppress() as prof:
            # Transfer data from CPU to GPU, compute on GPU, and transfer back to CPU
            start = time()
            model(data.to(device=new_device)).to(device="cpu")
            stop = time()
    except Exception as e:
        raise ValueError(
            f"Caught exception while attempting to profile model inference. "
            "Did you ensure that you model has a correct `input_shape` attribute (omitting the batch_size dimension)? "
            f"Exception contents: {e}"
        )
    finally:
        model.to(device=prev_device)
        if prevously_training:
            model.train()

    timing = stop - start  # seconds

    if detailed:
        return timing, prof
    return timing


def compute_num_runs(total_wait_time, single_run_time):
    additional_runs = 1
    single_run_time = max(single_run_time, 1e-7)
    for option in [10000, 1000, 100, 10, 5, 3]:
        if (total_wait_time / single_run_time / option) > 1:
            additional_runs = option
            break
    return additional_runs


def inference_timing(
    model: torch.nn.Module, max_wait_seconds=30, num_runs=None
) -> Tuple[Dict[str, str], torch.autograd.profiler.EventList]:
    for attr in ["hparams.batch_size"]:
        assert some(
            model, attr
        ), f"{name(model)} should define `{attr}` but none was found."

    # Make initial run to gauge its time. Also serves as a "dry-run"
    single_run_seconds, single_run_prof = time_single_inference(model, detailed=True)

    num_runs = (
        num_runs if num_runs else compute_num_runs(max_wait_seconds, single_run_seconds)
    )

    times_s = []
    for _ in tqdm(range(num_runs), desc="Profiling"):
        times_s.append(time_single_inference(model, detailed=False, warm_up=False))

    batch_size = model.hparams.batch_size
    s_per_sample = np.array(times_s) / batch_size
    samples_per_s = 1 / s_per_sample
    µs_per_sample = s_per_sample * 1e6

    results_dict = {}
    results_dict["micro_seconds_per_sample_mean"] = float(µs_per_sample.mean())
    results_dict["micro_seconds_per_sample_std"] = float(µs_per_sample.std())
    results_dict["samples_per_second_mean"] = float(samples_per_s.mean())
    results_dict["samples_per_second_std"] = float(samples_per_s.std())
    results_dict[
        "time_per_sample"
    ] = f"{format_time(µs_per_sample.mean())} +/- {format_time(µs_per_sample.std())} [{format_time(µs_per_sample.min())}, {format_time(µs_per_sample.max())}]"
    results_dict[
        "samples_per_second"
    ] = f"{samples_per_s.mean():.3f} +/- {samples_per_s.std():.3f} [{samples_per_s.min():.3f}, {samples_per_s.max():.3f}]"
    results_dict["num_runs"] = num_runs
    results_dict["on_gpu"] = (
        parse_num_gpus(model.hparams.gpus) > 0
        if hasattr(model.hparams, "gpus")
        else False
    )
    results_dict["batch_size"] = batch_size

    return results_dict, single_run_prof


class ProfileableDataset:
    @abstractmethod
    def profile(self) -> Dict[str, Any]:
        ...  # pragma: no cover


def format_bytes(bytes: int, prefix: str = None) -> str:
    power = 1024  # = 2 ** 10
    power_labels = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}

    n = 0
    while bytes > power:
        bytes /= power
        n += 1
    return f"{bytes:.3f} {power_labels[n]}B"


def format_bytes_fixed(bytes: int, prefix: str = "M") -> float:
    power = 1024  # = 2 ** 10
    labels_power = {"K": 1, "M": 2, "G": 3, "T": 4}
    prefix = prefix.upper()
    assert prefix in labels_power

    for _ in range(labels_power[prefix]):
        bytes /= power
    return bytes


def format_time(time_us):
    """Defines how to format time in FunctionEvent"""
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return "{:.3f}s".format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return "{:.3f}ms".format(time_us / US_IN_MS)
    return "{:.3f}us".format(time_us)
