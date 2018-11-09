import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )
#process.source.firstLuminosityBlock = cms.untracked.uint32(2)

process.options = cms.untracked.PSet(
#    numberOfThreads = cms.untracked.uint32(4),
    numberOfStreams = cms.untracked.uint32(0)
)
#process.Tracer = cms.Service("Tracer")

# Flow diagram of the modules
#
#     1   5
#    / \  |
#   2  4  6
#   |
#   3

process.load("HeterogeneousCore.CUDATest.prod1BeginJob_cff")
process.load("HeterogeneousCore.CUDATest.prod5BeginJob_cff")
process.load("HeterogeneousCore.CUDATest.prod6BeginJob_cff")

from HeterogeneousCore.ParameterSet.SwitchProducer import SwitchProducer

from HeterogeneousCore.CUDATest.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst
from HeterogeneousCore.CUDATest.testCUDAProducerGPU_cfi import testCUDAProducerGPU
from HeterogeneousCore.CUDATest.testCUDAProducerGPUEW_cfi import testCUDAProducerGPUEW
from HeterogeneousCore.CUDATest.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU

# GPU producers
process.prod2CUDA = testCUDAProducerGPU.clone(src = "prod1CUDA")
process.prod3CUDA = testCUDAProducerGPU.clone(src = "prod2CUDA")
process.prod4CUDA = testCUDAProducerGPUEW.clone(src = "prod1CUDA")

# Switches between CPU producer and the GPU-to-CPU transfer producers
from HeterogeneousCore.CUDATest.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod2 = SwitchProducer(
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod2CUDA"),
    cpu = testCUDAProducerCPU.clone(src = "prod1")
)
process.prod3 = SwitchProducer(
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod3CUDA"),
    cpu = testCUDAProducerCPU.clone(src = "prod2"),
)
process.prod4 = SwitchProducer(
    cuda = testCUDAProducerGPUtoCPU.clone(src = "prod4CUDA"),
    cpu = testCUDAProducerCPU.clone(src = "prod1")
)

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3_*_*",
        "keep *_prod4_*_*",
        "keep *_prod6_*_*",
    ),
                               verbosity = cms.untracked.uint32(2),
)
if False:
    process.out = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring(
            "keep *_prod3_*_*",
            "keep *_prod4_*_*",
            "keep *_prod6_*_*",
        ),
        fileName = cms.untracked.string("foo.root")
    )

process.tCUDA = cms.Task(
    process.prod2CUDA,
    process.prod3CUDA,
    process.prod4CUDA
)

process.tSwitch = cms.Task(
    process.prod2,
    process.prod3,
#    process.prod4 # test this one on a Path
)

process.t = cms.Task(
    process.prod1Task, process.prod5Task, process.prod6Task,
    process.tCUDA, process.tSwitch
)
process.p = cms.Path(
    process.prod4
)
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
