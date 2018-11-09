import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer

from HeterogeneousCore.CUDATest.prod6CPU_cfi import prod6CPU as _prod6CPU
from HeterogeneousCore.CUDATest.prod6CUDA_cfi import prod6CUDA
from HeterogeneousCore.CUDATest.prod6FromCUDA_cfi import prod6FromCUDA as _prod6FromCUDA

prod6 = SwitchProducer(
    cuda = _prod6FromCUDA.clone(),
    cpu = _prod6CPU.clone(src="prod5")
)

prod6TaskCUDA = cms.Task(prod6CUDA)

prod6Task = cms.Task(
    prod6TaskCUDA,
    prod6
)
