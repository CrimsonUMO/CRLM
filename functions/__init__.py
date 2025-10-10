# upfe/__init__.py

from .utils import show_side_by_side  # 导入辅助功能
from .extractor import feature_extractor  # 导入特征提取函数

__all__ = [
    "show_side_by_side",
    "feature_extractor",
]