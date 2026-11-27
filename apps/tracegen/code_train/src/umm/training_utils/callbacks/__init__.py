"""PyTorch Lightning callbacks for the UMM training pipeline.

Provides custom callbacks for console metric printing, periodic visualization,
and rich-formatted model summaries. These are assembled by
``umm.training_utils.trainer_helper.get_trainer`` and attached to the
``pytorch_lightning.Trainer``.

Exported callbacks:
    - ``PrintingCallback``: Time-throttled console metric logging.
    - ``VizCallback``: Periodic ``pl_module.viz()`` invocation.
    - ``RichModelSummaryLoggingCallback``: Rich table model summary with file
      logging.
    - ``TgEvalCallback``: TraceGen reference evaluation at step cadence
      
"""

from .printing import PrintingCallback
from .viz import VizCallback
from .modelsummary import RichModelSummaryLoggingCallback
from .tg_eval_callback import TgEvalCallback
from .tg128_eval_callback import Tg128EvalCallback
from .tg128_sliding_viz_callback import Tg128SlidingVizCallback
