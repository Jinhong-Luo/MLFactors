"""因子抽象基类。"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class BaseFactor(ABC):
    """选股因子基类。

    Attributes
    ----------
    name : 因子唯一标识名
    description : 因子描述
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        constituents: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        """生成选股信号矩阵。

        Parameters
        ----------
        data : (date, symbol) MultiIndex 数据

        Returns
        -------
        pd.DataFrame，索引为 date，列为 symbol，值为因子信号
        """
        ...
