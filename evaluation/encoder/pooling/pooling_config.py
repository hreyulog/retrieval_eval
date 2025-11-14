from .pooling_cls import Pooling
from .mean_pooling import MeanPooling
from .cls_pooling import ClsPooling

_FACTORIES= {
    MeanPooling.code_name():MeanPooling,
    ClsPooling.code_name():ClsPooling
}


def create_pooling(name: str) -> Pooling:
    return _FACTORIES[name]()