"""训练过程回调：显存回收与按 epoch 保存检查点。"""

from __future__ import annotations

from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import TrainerControl, TrainerState

from phi2_mini.logging_utils import get_logger

try:  # 允许在无 torch 环境下导入本模块（CI 里跑纯逻辑测试）
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = get_logger(__name__)


class EmptyCacheCallback(TrainerCallback):
    """每 N 次日志清一次 CUDA 缓存，缓解低显存设备上的显存缓慢增长。"""

    def __init__(self, every_n_logs: int = 2) -> None:
        self.every_n_logs = every_n_logs
        self.log_count = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.log_count += 1
        if torch is not None and self.every_n_logs > 0 and self.log_count % self.every_n_logs == 0:
            torch.cuda.empty_cache()


class SaveOnEpochEndCallback(TrainerCallback):
    """每个 epoch 结束时保存一次模型。

    ``TrainingArguments`` 的 ``save_strategy`` 中 epoch 与 steps 不兼容，
    这里通过 ``control.should_save`` 手动触发保存。
    """

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        control.should_save = True
        return control


class DefaultTrainerCallback(EmptyCacheCallback, SaveOnEpochEndCallback):
    """SFT/DPO 默认回调：等价于重构前的 ``MyTrainerCallback``。"""


# 兼容重构前的类名，下一个版本移除
MyTrainerCallback = DefaultTrainerCallback
