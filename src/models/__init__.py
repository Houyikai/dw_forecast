__all__ = ['MLP', 'NHITS', 'PatchTST', 'LSTM']

#四种模型：

# LSTM 简单 长时间预测性能差
# MLP 简单 长时间预测性能差
# NHITS 适合长时间预测
# PatchTST 适合长时间预测

from .lstm import LSTM
from .mlp import MLP
from .nhits import NHITS
from .patchtst import PatchTST