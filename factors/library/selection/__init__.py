"""内置选股因子库。

导入各子模块以触发 @register_factor 装饰器，将因子注册到 FactorRegistry。
"""

from . import (
    alpha158_kline,
    alpha158_price,
    alpha158_rolling,
    capm_beta,
    ordinal_factor_rotation,
    quality_combine,
    quality_growth,
    svm_alpha158,
    volatility,
)  # noqa: F401
