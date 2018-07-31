import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerFallback_cfi import testCUDAProducerFallback as _testCUDAProducerFallback
prod5Fallback = _testCUDAProducerFallback.clone()
