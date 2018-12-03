import os
import pickle

import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer as _SwitchProducer

def _switch_cuda():
    return (os.system("edmCUDAEnabled") == 0, 2)

class SwitchProducerCUDA(_SwitchProducer):
    def __init__(self, **kargs):
        super(SwitchProducerCUDA,self).__init__(
            dict(
                cpu = _SwitchProducer.getCpu,
                cuda = SwitchProducerCUDA.getCuda()
            ), **kargs)

    @staticmethod
    def getCuda():
        return _switch_cuda


if __name__ == "__main__":
    import unittest

    class testSwitchProducerCUDA(unittest.TestCase):
        def testPickle(self):
            p = cms.Process("Test")
            p.sp = SwitchProducerCUDA(
                cpu = cms.EDProducer("A"),
                cuda = cms.EDProducer("B")
            )
            pkl = pickle.dumps(p)
            unpkl = pickle.loads(pkl)
            self.assertEqual(unpkl.sp.cpu.type_(), "A")
            self.assertEqual(unpkl.sp.cuda.type_(), "B")

    unittest.main()
