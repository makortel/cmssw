import FWCore.ParameterSet.Config as cms

process = cms.Process("Test")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

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

process.load("HeterogeneousCore.CUDATest.prod1_cff")
process.load("HeterogeneousCore.CUDATest.prod5_cff")
process.load("HeterogeneousCore.CUDATest.prod6_cff")

# CPU producers
from HeterogeneousCore.CUDATest.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod2CPU = testCUDAProducerCPU.clone(src = "prod1CPU")
process.prod3CPU = testCUDAProducerCPU.clone(src = "prod2CPU")
process.prod4CPU = testCUDAProducerCPU.clone(src = "prod1CPU")

from HeterogeneousCore.CUDATest.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst
from HeterogeneousCore.CUDATest.testCUDAProducerGPU_cfi import testCUDAProducerGPU
from HeterogeneousCore.CUDATest.testCUDAProducerGPUEW_cfi import testCUDAProducerGPUEW
from HeterogeneousCore.CUDATest.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU

# GPU producers
process.prod2CUDA = testCUDAProducerGPU.clone(src = "prod1CUDA")
process.prod3CUDA = testCUDAProducerGPU.clone(src = "prod2CUDA")
process.prod4CUDA = testCUDAProducerGPUEW.clone(src = "prod1CUDA")

# Modules to copy data from GPU to CPU (as "on demand" as any other
# EDProducer, i.e. according to consumes() and prefetching)
process.prod2FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod2CUDA")
process.prod3FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod3CUDA")
process.prod4FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod4CUDA")

# These ones are to provide backwards compatibility to the downstream
# clients. To be replaced with an enhanced version of EDAlias (with an
# ordered fallback mechanism).
from HeterogeneousCore.CUDATest.testCUDAProducerFallback_cfi import testCUDAProducerFallback
process.prod2 = testCUDAProducerFallback.clone(src = ["prod2FromCUDA", "prod2CPU"])
process.prod3 = testCUDAProducerFallback.clone(src = ["prod3FromCUDA", "prod3CPU"])
process.prod4 = testCUDAProducerFallback.clone(src = ["prod4FromCUDA", "prod4CPU"])

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3_*_*",
        "keep *_prod4_*_*",
        "keep *_prod5_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)

process.prodCPU1 = cms.Path(
    ~process.prod1CUDADeviceFilter +
    process.prod2CPU +
    process.prod3CPU +
    process.prod4CPU
)
process.prodCUDA1 = cms.Path(
    process.prod1CUDADeviceFilter +
    process.prod2CUDA +
    process.prod2FromCUDA +
    process.prod3CUDA +
    process.prod3FromCUDA +
    process.prod4CUDA +
    process.prod4FromCUDA
)

process.t = cms.Task(
    # Eventually the goal is to specify these as part of a Task,
    # but (at least) as long as the fallback mechanism is implemented
    # with an EDProducer, they must be in a Path.
#    process.prod2CPU, process.prod3CPU, process.prod4CPU,
#    process.prod2CUDA, process.prod3CUDA, process.prod4CUDA,
#    process.prod2FromCUDA, process.prod3FromCUDA, process.prod4FromCUDA,

    process.prod2, process.prod3, process.prod4,
    process.prod1Task, process.prod5Task, process.prod6Task
)
process.p = cms.Path()
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
