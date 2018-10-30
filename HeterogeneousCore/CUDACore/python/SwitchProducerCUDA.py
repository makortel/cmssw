import os

from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer

def availableResourcesCUDA():
    ret = []
    if os.system("edmCUDAEnabled") == 0:
        ret.append("cuda")
    ret.append("cpu")
    return ret

class SwitchProducerCUDA(SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerCUDA, self).__init__(availableResourcesCUDA, **kargs)
