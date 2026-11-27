"""Configuration parsing utilities for UMM training.

Defines the CLI argument parser (via ``configargparse``), a dict-like
``Config`` class that supports attribute access and recursive nesting,
and the main ``get_config`` entry point that merges YAML config files
with command-line overrides.

Example:
    >>> from umm.training_utils.config_helpers import get_config
    >>> cfg = get_config(["--config", "configs/maniskill/ms1.0_pickcube.yaml"])
    >>> cfg.expname
    'experiment'
"""

import configargparse


def _float_or_int(value: str):
    """Parse as int if whole number > 1, else float. Matches Lightning's limit_train_batches semantics."""
    f = float(value)
    if f == int(f) and f > 1:
        return int(f)
    return f


def config_parser():
    """Build the ``configargparse`` argument parser with all predefined training flags.

    The parser defines arguments for experiment naming, GPU selection, model
    resumption, precision, seeding, batch sizes, gradient clipping, validation
    scheduling, logging intervals, visualization callbacks, checkpoint saving,
    and console printing behaviour.

    Returns:
        configargparse.ArgumentParser: The fully-configured argument parser.

    Example:
        >>> parser = config_parser()
        >>> args, unknown = parser.parse_known_args(["--expname", "test"])
        >>> args.expname
        'test'
    """
    parser = configargparse.ArgumentParser(
        description="Jiahui's new 2023 codebase from PyTorch Lightning",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", is_config_file=True, help="config file path")
    parser.add_argument(
        "--expname", type=str, help="experiment name", default="experiment"
    )
    parser.add_argument(
        "--wandb_name_postfix", type=str, help="wandb name postfix", default=""
    )
    parser.add_argument("--log_dir", type=str, help="experiment name", default="./logs")

    # ! TODO: distributed dev and gpu stuff
    parser.add_argument("--gpu", type=str, help="gpus to use", default="all")
    # parser.add_argument("--distributed", help="TODO: dev this", default=False, action="store_true")

    parser.add_argument(
        "--enforce_local_provider",
        help="enforce to load from /gcs local file system",
        default=False,
        action="store_true",
    )

    # model resume
    parser.add_argument(
        "--resume",
        "-r",
        help=r"""
        resume behavior: -r have to pass in a value
        1.) if the arg is a version number, resume that version with the latest ckpt if that version exists; otherwise use this version number to create a new exp log;
        2.) else if the arg is a ckpt path and it exists, identify the version from the ckpt path and resume that version with that ckpt
        """,
        type=str,
        default=None,
    )

    # precision
    parser.add_argument(
        "--precision",
        "-p",
        help="precision, 16 or 32",
        choices=[16, 32],
        default=32,
        type=int,
    )
    parser.add_argument(
        "--matmul_precision",
        "-m",
        help="For tensor cores boost, set: torch.set_float32_matmul_precision(args)",
        default="highest",
        choices=["highest", "high", "medium"],
    )

    parser.add_argument(
        "--init_seed",
        help="seed used for init, warning: if resume, this may not work at this moment",
        default=None,
    )

    parser.add_argument(
        "--workers", "-w", help="number of dataloader workers", default=0, type=int
    )

    parser.add_argument(
        "--max_steps",
        "-n",
        help="number of global iterations",
        default=int(1e9),
        type=int,
    )
    parser.add_argument(
        "--max_epochs",
        "-e",
        help="number of max global epoches",
        default=int(1e6),
        type=int,
    )

    parser.add_argument(
        "--gradient_clip_val",
        help="set trainer `gradient_clip_val`",
        default=None,
        type=float,
    )
    parser.add_argument(
        "--gradient_clip_algorithm",
        help="set trainer `gradient_clip_algorithm`",
        default="norm",
        type=str,
        choices=["norm", "value"],
    )

    parser.add_argument(
        "--train_batchsize",
        "-tb",
        help="batchsize for train phase",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--eval_batchsize",
        "-eb",
        help="batchsize for non-train phase",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        "-agb",
        help="accumulate_grad_batches",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--limit_train_batches",
        help="float in (0,1] = fraction of training set, int > 1 = number of batches per epoch",
        default=1.0,
        type=_float_or_int,
    )
    parser.add_argument(
        "--limit_val_batches",
        help="[int] number of val step per val loop",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--fast_dev_run",
        help="[int] run N train+val batches then stop (disables logging/checkpointing). 0 = off",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        help="[int] set `num_sanity_val_steps` in trainer",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--anomaly",
        "-a",
        dest="enable_anomaly",
        default=False,
        action="store_true",
        help="(Bool) If set, use pytorch anomaly detection and nan, inf hook",
    )
    parser.add_argument(
        "--debug",
        "-d",
        dest="debug_logging",
        default=False,
        action="store_true",
        help="(Bool) If set, set logging level to DEBUG, otherwise INFO",
    )

    # val
    parser.add_argument(
        "--val_every_n_iter",
        help="perform the val loop every N train iter, set `val_check_interval` in trainer",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--val_every_n_epoch",
        help="perform the val loop every N epoch, set `check_val_every_n_epoch` in trainer",
        default=None,
        type=int,
    )

    # log
    parser.add_argument(
        "--log_every_n_steps",
        help="tb logger will log every N steps, set `log_every_n_steps`",
        default=50,
        type=int,
    )
    parser.add_argument(
        "--log_organize_subfolder_interval",
        help="tb logger will organize the images, meshes saving to subfolders every N global steps",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--progress_bar_refresh_rate",
        "-f",
        help="set `progress_bar_refresh_rate` in progress bar callback",
        default=1,
        type=float,
    )

    # callbacks
    # * viz
    parser.add_argument(
        "--viz_every_n_train_steps",
        help="call pl_model.viz() function every N train steps, if set to -1, won't call",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--viz_every_n_val_steps",
        help="call pl_model.viz() function every N va; steps, if set to -1, won't call",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--viz_every_n_test_steps",
        help="call pl_model.viz() function every N test steps, if set to -1, won't call",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--viz_try",
        help="if set, won't raise error if any interupt happens during viz",
        default=False,
        action="store_true",
    )

    # * ckpt callback
    parser.add_argument(
        "--ckpt_monitor",
        help="the name of metric the ckpt callback will monitor",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--ckpt_monitor_mode",
        help="the ckpt callback will select the min or max of the metric",
        default="min",
        type=str,
        choices=["min", "max"],
    )
    parser.add_argument(
        "--ckpt_save_top_k",
        help="if monitor is set, save top k ckpt",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--ckpt_every_n_train_steps",
        help="save checkpoint every N train steps",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--ckpt_warmup_steps",
        help=(
            "skip topk monitor-based saves while global_step < this value. "
            "Use when the ckpt_monitor metric is logged at a slower cadence "
            "than ckpt_every_n_train_steps (e.g. eval-only metrics). Set to "
            "(first eval step + small margin), e.g. 11000 if eval fires every "
            "10000 steps. last.ckpt is unaffected. Default 0 = no warmup."
        ),
        default=0,
        type=int,
    )
    parser.add_argument(
        "--ckpt_every_n_epochs",
        help="save checkpoint every N epochs",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--ckpt_save_on_train_epoch_end",
        help="if set, will also save ckpt on train epoch end, otherwise do the ckpt at the end of val loop by default (because often you want to monitor the val metrics, not the training metric)",
        default=False,
        action="store_true",
    )

    # * printing callback
    parser.add_argument(
        "--train_print_freq_seconds",
        help="console print freq, print every N seconds",
        default=3.0,
        type=float,
    )
    parser.add_argument(
        "--nontrain_print_every_step",
        help="if set, will print in console every val,test,predict step",
        default=False,
        action="store_true",
    )

    # * shortcut data change
    parser.add_argument(
        "--data_dir",
        help="quick console set for dataset root, used in datamodule",
        default="./data",
        type=str,
    )
    return parser


class Config(dict):
    """A dict subclass that supports attribute-style access and recursive nesting.

    Nested plain ``dict`` values are automatically converted to ``Config``
    instances on insertion, enabling dot-notation access at any depth.

    Example:
        >>> cfg = Config({"model": {"lr": 0.001}})
        >>> cfg.model.lr
        0.001
        >>> cfg.model.lr = 0.01
        >>> cfg["model"]["lr"]
        0.01
    """

    def __init__(self, *args, **kwargs):
        """Initialise from any mapping/iterable accepted by ``dict``.

        Args:
            *args: Positional arguments forwarded to ``dict.__init__``.
            **kwargs: Keyword arguments forwarded to ``dict.__init__``.
        """
        super().__init__(*args, **kwargs)
        # Recursively convert initial values
        for k, v in self.items():
            super().__setitem__(k, self._convert(v))

    def _convert(self, v):
        """Recursively wrap ``dict`` values as ``Config`` and process lists.

        Args:
            v: The value to convert.

        Returns:
            The converted value (``Config`` for dicts, recursively processed
            lists, or the original value for other types).
        """
        if isinstance(v, dict) and not isinstance(v, Config):
            return Config(v)
        if isinstance(v, list):
            return [self._convert(x) for x in v]
        return v

    def __setitem__(self, key, value):
        """Set an item, converting ``dict`` values to ``Config``.

        Args:
            key: The key to set.
            value: The value to store (dicts are auto-converted).
        """
        super().__setitem__(key, self._convert(value))

    def __getattr__(self, key):
        """Access dict keys as attributes.

        Args:
            key: The attribute/key name.

        Returns:
            The value stored under *key*.

        Raises:
            AttributeError: If *key* does not exist in the dict.
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        """Set dict keys via attribute assignment.

        Args:
            key: The attribute/key name.
            value: The value to store.
        """
        self[key] = value

    def update(self, *args, **kwargs):
        """Update the config, converting values as needed.

        Accepts the same argument forms as ``dict.update``: a single mapping,
        an iterable of key-value pairs, and/or keyword arguments.

        Args:
            *args: A mapping or iterable of key-value pairs (at most one).
            **kwargs: Additional key-value pairs.

        Raises:
            TypeError: If more than one positional argument is provided.
        """
        if args:
            if len(args) > 1:
                raise TypeError(
                    "update expected at most 1 arguments, got %d" % len(args)
                )
            other = args[0]
            if isinstance(other, dict):
                for k, v in other.items():
                    self[k] = v
            elif hasattr(other, "keys"):
                for k in other.keys():
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v


def convert_flexible_args(args: list):
    """Convert unknown args from command line to nested dict structure.

    Supports:
    - Simple args: --key=value
    - Nested args: --model.lr=0.001 -> {"model": {"lr": 0.001}}
    - List args: --layers=[1,2,3]
    - Boolean flags: --use_feature (without =value)

    Args:
        args: List of raw CLI argument strings (e.g.
            ``["--model.lr=0.001", "--debug"]``).

    Returns:
        Config: A ``Config`` object with parsed and nested values.

    Example:
        >>> convert_flexible_args(["--model.lr=0.001", "--debug"])
        Config({'model': Config({'lr': 0.001}), 'debug': True})
    """
    ret = {}
    for line in args:
        if "=" in line:
            # Parse key=value format
            key_part = line.split("=")[0].strip("--").strip("-")
            value_part = "=".join(line.split("=")[1:])

            # Try to evaluate the value (handles numbers, lists, dicts, bools, None)
            try:
                v = eval(value_part)
            except Exception:
                # If eval fails, treat as string
                v = value_part
        else:
            # Boolean flag without value (e.g., --debug)
            key_part = line.strip("--").strip("-")
            v = True

        # Handle nested keys (e.g., "model.lr" -> {"model": {"lr": ...}})
        if "." in key_part:
            keys = key_part.split(".")
            current = ret
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = v
        else:
            ret[key_part] = v

    ret = Config(ret)
    return ret


def get_config(args_str=None):
    """Parse config from command line and config file.

    Priority (highest to lowest):
    1. Command line args (both predefined and custom)
    2. Config file (loaded via --config)

    All command line args (whether predefined or not) will override config file values.

    Args:
        args_str: Optional list of argument strings to parse instead of
            ``sys.argv``. Useful for programmatic invocation and testing.

    Returns:
        Config: A ``Config`` object containing the merged configuration.

    Example:
        >>> cfg = get_config(["--config", "configs/ms1.0.yaml", "--model.lr=0.001"])
        >>> cfg.expname
        'experiment'
    """
    # read cmd flags
    parser = config_parser()
    predefined, customized = parser.parse_known_args(args=args_str)

    # Convert predefined args to dict
    predefined_dict = vars(predefined)

    # Parse custom/unknown args (these may be model configs, dataset configs, etc.)
    custom_cfg = convert_flexible_args(customized)

    # Merge: Start with custom configs, then override with predefined training args
    # This way, command line args always take precedence
    cfg = custom_cfg
    cfg.update(predefined_dict)

    return cfg  # a Config object, an object like dict
