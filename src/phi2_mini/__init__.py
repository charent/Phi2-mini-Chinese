"""Phi2-mini-Chinese：从零训练 0.2B 中文 Phi2 小模型。

包内模块按职责划分：

- ``phi2_mini.data.*``   数据集构建 / 分词映射 / 去重
- ``phi2_mini.modeling.*`` 模型配置与 attention 后端探测
- ``phi2_mini.train.*``  预训练 / SFT / DPO 训练流程
- ``phi2_mini.inference`` 推理与 web demo
"""

__version__ = '0.2.0'

__all__ = ['__version__']
