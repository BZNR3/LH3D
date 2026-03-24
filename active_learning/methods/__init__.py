from .random import RandomSelector
from .entropy import EntropySelector
from .LH3D import LH3DSelector


_METHOD_REGISTRY = {
    "random": RandomSelector,
    "entropy": EntropySelector,
    "lh3d": LH3DSelector,

}

def get_method(name: str):
    name = (name or "uncertainty").lower()
    if name not in _METHOD_REGISTRY:
        raise KeyError(f"Unknown active method '{name}'. Options: {list(_METHOD_REGISTRY.keys())}")
    return _METHOD_REGISTRY[name]