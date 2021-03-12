import logging
import socket
import subprocess
from functools import wraps
from os import makedirs
from pathlib import Path
from typing import Callable

import click
import coloredlogs
import pytorch_lightning as pl

from ride.utils.env import LOG_LEVEL


def _process_rank():
    if pl.utilities._HOROVOD_AVAILABLE:
        import horovod.torch as hvd

        hvd.init()
        return hvd.rank()
    else:
        return pl.utilities.rank_zero_only.rank


process_rank = _process_rank()


def once(fn: Callable):
    mem = set()

    @wraps(fn)
    def wrapped(*args, **kwargs):
        h = hash((args, str(kwargs)))
        if h in mem:
            return
        else:
            mem.add(h)
            return fn(*args, **kwargs)

    return wrapped


def if_rank_zero(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global process_rank
        if process_rank == 0:
            fn(*args, **kwargs)

    return wrapped


def getLogger(name, log_once=False):
    name = name.split(".")[0]  # Get chars before '.'
    if name not in {"wandb", "lightning", "ride", "datasets", "models"}:
        name = click.style(name, fg="white", bold=True)
    logger = logging.getLogger(name)
    if log_once:
        logger._log = once(logger._log)
    logger._log = if_rank_zero(logger._log)
    return logger


logger = getLogger(__name__)


def style_logging():

    LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    assert LOG_LEVEL in LOG_LEVELS, f"Specified LOG_LEVEL should be one of {LOG_LEVELS}"

    coloredlogs.install(
        level=LOG_LEVEL,
        fmt="%(name)s: %(message)s",
        level_styles={
            "debug": {"color": "white", "faint": True},
            "warning": {"bold": True},
            "error": {"color": "red", "bold": True},
        },
    )

    # Block pytorch_lightning from writing directly to stdout
    lightning_logger = getattr(pl, "_logger", logging.getLogger("lightning"))
    lightning_logger.handlers = []
    lightning_logger.propagate = bool(process_rank == 0)

    # Set coloring
    lightning_logger.name = click.style(lightning_logger.name, fg="yellow", bold=True)

    ride_logger = logging.getLogger("ride")
    ride_logger.name = click.style(ride_logger.name, fg="cyan", bold=True)

    datasets_logger = logging.getLogger("datasets")
    datasets_logger.name = click.style(datasets_logger.name, fg="magenta", bold=True)

    models_logger = logging.getLogger("models")
    models_logger.name = click.style(models_logger.name, fg="green", bold=True)

    # Block matplotlib debug logger
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(
        logging.INFO if LOG_LEVEL == "DEBUG" else getattr(logging, LOG_LEVEL)
    )


def init_logging(logdir: str = None, logging_backend: str = "tensorboard"):
    if not logdir:
        return

    # Add root handler for redirecting run output to file
    makedirs(logdir, exist_ok=True)
    getLogger("").addHandler(logging.FileHandler(Path(logdir) / "run.log"))

    # Write basic environment info to logs
    logger.info(f"Running on host {click.style(socket.gethostname(), fg='yellow')}")

    try:
        git_repo = (
            subprocess.check_output(
                "git config --get remote.origin.url",
                shell=True,
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8")
            .strip()
        )
        git_tag = (
            subprocess.check_output(
                "git rev-parse HEAD", shell=True, stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        git_msg = click.style(
            f"{git_repo.replace('.git','')}/tree/{git_tag}", fg="blue", bold=False
        )
        logger.info(f"⭐️ View project repository at {git_msg}")
    except subprocess.CalledProcessError:
        pass

    logger.info(f"Run data is saved locally at {logdir}")
    logger.info(
        f"Logging using {click.style(logging_backend.capitalize(), fg='yellow')}"
    )
