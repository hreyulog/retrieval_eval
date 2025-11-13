from geomas.encoder.pooling.pooling import Pooling
from geomas.encoder.pooling.mean_pooling import MeanPooling
from geomas.encoder.pooling.cls_pooling import ClsPooling

_FACTORIES= {
    MeanPooling.code_name():MeanPooling,
    ClsPooling.code_name():ClsPooling
}


def create_pooling(name: str) -> Pooling:
    return _FACTORIES[name]()