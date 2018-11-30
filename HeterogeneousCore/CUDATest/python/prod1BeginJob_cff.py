import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

from HeterogeneousCore.CUDATest.prod1CUDADevice_cfi import prod1CUDADevice
from HeterogeneousCore.CUDATest.prod1CPU_cfi import prod1CPU as _prod1CPU
from HeterogeneousCore.CUDATest.prod1CUDA_cfi import prod1CUDA
from HeterogeneousCore.CUDATest.prod1FromCUDA_cfi import prod1FromCUDA as _prod1FromCUDA

prod1CUDA.src = "prod1CUDADevice"

prod1 = SwitchProducerCUDA(
    cuda = _prod1FromCUDA.clone(),
    cpu = _prod1CPU.clone()
)

prod1TaskCUDA = cms.Task(prod1CUDADevice, prod1CUDA)

prod1Task = cms.Task(
    prod1TaskCUDA,
    prod1
)
