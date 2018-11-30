from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer as _SwitchProducer

import os

class SwitchProducerCUDA(_SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerCUDA,self).__init__(
            dict(
                cpu = _SwitchProducer.cpu,
                cuda = lambda: (os.system("edmCUDAEnabled") == 0, 2)
            ), **kargs)
