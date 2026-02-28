from __future__ import annotations  # 使用前瞻特性，使类型注解支持字符串和未来类型

from typing import Optional  # 可选类型提示
from transformers import get_linear_schedule_with_warmup  # transformers库中的线性warmup调度器


def build_scheduler(
    optimizer,  # 传入的优化器
    num_training_steps: int,  # 总训练步数 (可能包含梯度累积后的实际步数)
    warmup_ratio: float = 0.1,  # 热身步数比例，默认0.1
    num_warmup_steps: Optional[int] = None,  # 明确指定热身步数则覆盖比例
):
    """
    Build a learning rate scheduler.

    Default: linear warmup + linear decay (most common for BERT fine-tuning)

    Args:
        optimizer: torch optimizer
        num_training_steps: total optimization steps (after gradient accumulation)
        warmup_ratio: warmup steps ratio in [0, 1]
        num_warmup_steps: if provided, override warmup_ratio

    Returns:
        scheduler (from transformers)
    """
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0, got {num_training_steps}")  # 验证步数正数

    if num_warmup_steps is None:
        if warmup_ratio < 0 or warmup_ratio > 1:
            raise ValueError(f"warmup_ratio must be in [0,1], got {warmup_ratio}")  # 检查比例合法性
        num_warmup_steps = int(num_training_steps * warmup_ratio)  # 计算热身步数

    # At least 0 warmup steps
    num_warmup_steps = max(0, int(num_warmup_steps))  # 确保热身步数非负

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,  # 传入的优化器，用于更新学习率
        num_warmup_steps=num_warmup_steps,  # 设置热身阶段步数
        num_training_steps=num_training_steps,  # 总训练步数，用于线性衰减
    )  # 创建线性warmup + 线性衰减的调度器
    return scheduler  # 返回创建的调度器实例