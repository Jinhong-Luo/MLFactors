"""因子注册表。"""

from factors.base import BaseFactor


class FactorRegistry:
    """注册、获取和列出因子类。"""

    _registry: dict[str, type[BaseFactor]] = {}

    @classmethod
    def register(cls, factor_class: type[BaseFactor]) -> type[BaseFactor]:
        if not issubclass(factor_class, BaseFactor):
            raise TypeError("注册对象必须是 BaseFactor 子类")
        if not factor_class.name:
            raise ValueError(f"{factor_class.__name__} 必须定义非空的 name 属性")
        cls._registry[factor_class.name] = factor_class
        return factor_class

    @classmethod
    def get(cls, name: str) -> type[BaseFactor]:
        return cls._registry[name]

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)


def register_factor(factor_class: type[BaseFactor]) -> type[BaseFactor]:
    """将 BaseFactor 子类注册到全局注册表。"""
    return FactorRegistry.register(factor_class)
