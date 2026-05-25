"""内置选股因子库。

导入各子模块以触发 @register_factor 装饰器，将因子注册到 FactorRegistry。
"""

from . import (
    alpha158_kline,
    alpha158_price,
    alpha158_rolling,
    momentum,
    ordinal_factor_rotation,
    volatility,
)  # noqa: F401
