import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod1CUDADeviceFilter_cfi import prod1CUDADeviceFilter
from HeterogeneousCore.CUDATest.prod1CPU_cfi import prod1CPU
from HeterogeneousCore.CUDATest.prod1CUDA_cfi import prod1CUDA
from HeterogeneousCore.CUDATest.prod1FromCUDA_cfi import prod1FromCUDA

from HeterogeneousCore.CUDATest.testCUDAProducerFallback_cfi import testCUDAProducerFallback as _testCUDAProducerFallback

prod1 = _testCUDAProducerFallback.clone(src = ["prod1CUDA", "prod1CPU"])

prod1PathCUDA = cms.Path(
    prod1CUDADeviceFilter +
    prod1CUDA
)
prod1PathCPU = cms.Path(
    ~prod1CUDADeviceFilter +
    prod1CPU
)

prod1Task = cms.Task(
    prod1FromCUDA, prod1
)
