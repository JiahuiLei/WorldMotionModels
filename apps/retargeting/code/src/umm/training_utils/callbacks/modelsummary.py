"""Rich-formatted model summary callback with file logging.

Extended from ``pytorch_lightning.callbacks.RichModelSummary`` to also
write the plain-text summary table to the Python ``logging`` file handler
(without duplicating it to the console).

Example:
    >>> cb = RichModelSummaryLoggingCallback(max_depth=1)
    >>> trainer = pl.Trainer(callbacks=[cb])
"""

"""Rich model summary callback with file-logging support.

Extends Lightning's ModelSummary to render a rich-formatted table of
model layers to the console and also log the plain-text summary to the
file logger (bypassing the console handler to avoid duplication).

Derived from the Lightning AI RichModelSummary source.
"""

# From RichModelSummary,add logging
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Tuple

from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.progress.rich_progress import _RICH_AVAILABLE
from pytorch_lightning.utilities.model_summary import get_human_readable_count
import logging
from pytorch_lightning.utilities.model_summary.model_summary import (
    _format_summary_table,
)

if _RICH_AVAILABLE:  # type: ignore[has-type]
    from rich import get_console
    from rich.table import Table

from pytorch_lightning.utilities import rank_zero_only


class RichModelSummaryLoggingCallback(ModelSummary):
    r"""
    Generates a summary of all layers in a :class:`~pytorch_lightning.core.module.LightningModule`
    with `rich text formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichModelSummary

        trainer = Trainer(callbacks=RichModelSummary())

    You could also enable ``RichModelSummary`` using the :class:`~pytorch_lightning.callbacks.RichProgressBar`

    .. code-block:: python

        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import RichProgressBar

        trainer = Trainer(callbacks=RichProgressBar())

    Args:
        max_depth: The maximum depth of layer nesting that the summary will include. A value of 0 turns the
            layer summary off.

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.
    """

    def __init__(self, max_depth: int = 1) -> None:
        """Initialize the rich model summary callback.

        Args:
            max_depth: Maximum depth of layer nesting to include. A value
                of 0 disables the layer summary.

        Raises:
            ModuleNotFoundError: If the rich package is not installed.

        Example:
            >>> callback = RichModelSummaryLoggingCallback(max_depth=2)
        """
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichModelSummary` requires `rich` to be installed. Install it by running `pip install -U rich`."
            )
        super().__init__(max_depth)

    @staticmethod
    @rank_zero_only
    def summarize(
        summary_data: List[Tuple[str, List[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs,
    ) -> None:
        """Print a rich-formatted model summary table and log it to file.

        Renders the summary as a Rich table to the console, then writes a
        plain-text version to the file logger (temporarily removing the
        console handler to avoid duplicate output).

        Args:
            summary_data: List of (column_name, column_values) tuples
                describing the model layers.
            total_parameters: Total number of model parameters.
            trainable_parameters: Number of trainable parameters.
            model_size: Estimated model size in MB.
            total_training_modes: Dict mapping training mode names to counts.
            **summarize_kwargs: Additional keyword arguments (unused).

        Example:
            >>> RichModelSummaryLoggingCallback.summarize(
            ...     summary_data, total_params, train_params, size, modes)
        """
        console = get_console()

        table = Table(header_style="bold magenta")
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")

        column_names = list(zip(*summary_data))[0]

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data)))
        for row in rows:
            table.add_row(*row)

        console.print(table)
        try:
            summary_table = _format_summary_table(
                total_parameters,
                trainable_parameters,
                model_size,
                total_training_modes,
                *summary_data,
            )
        except TypeError:
            # PL 2.6+ changed _format_summary_table signature (added total_flops)
            # Fall back to a simple summary string
            summary_table = (
                f"\nTotal params: {get_human_readable_count(total_parameters)}\n"
                f"Trainable params: {get_human_readable_count(trainable_parameters)}\n"
                f"Non-trainable params: {get_human_readable_count(total_parameters - trainable_parameters)}\n"
                f"Model size: {model_size:.1f} MB\n"
            )

        # disable logging console printing
        # ! this logging tricky may not be totally safe...
        logger = logging.getLogger()  # get rot handler
        console_handler = None
        for handler in logger.handlers:
            if (
                isinstance(handler, logging.StreamHandler)
                and "<std" in handler.stream.name
            ):
                console_handler = handler
                break
        if console_handler is not None:
            logger.removeHandler(console_handler)
        logger.info(summary_table)
        # add back the console handler
        if console_handler is not None:
            logger.addHandler(console_handler)

        parameters = []
        for param in [
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size,
        ]:
            parameters.append("{:<{}}".format(get_human_readable_count(int(param)), 10))

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(
            f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}"
        )

        console.print(grid)
