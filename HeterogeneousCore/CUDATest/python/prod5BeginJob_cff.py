import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer

from HeterogeneousCore.CUDATest.prod5CUDADevice_cfi import prod5CUDADevice
from HeterogeneousCore.CUDATest.prod5CPU_cfi import prod5CPU as _prod5CPU
from HeterogeneousCore.CUDATest.prod5CUDA_cfi import prod5CUDA
from HeterogeneousCore.CUDATest.prod5FromCUDA_cfi import prod5FromCUDA as _prod5FromCUDA

prod5CUDA.src = "prod5CUDADevice"

prod5 = SwitchProducer(
    cuda = _prod5FromCUDA.clone(),
    cpu = _prod5CPU.clone()
)

prod5Task = cms.Task(
    prod5CUDADevice, prod5CUDA, prod5
)
