import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.testCUDAProducerFallback_cfi import testCUDAProducerFallback as _testCUDAProducerFallback
prod6Fallback = _testCUDAProducerFallback.clone()
