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
#    / \
#   2  4
#   |
#   3
#
# CPU producers
from HeterogeneousCore.CUDACore.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod1CPU = testCUDAProducerCPU.clone()
process.prod2CPU = testCUDAProducerCPU.clone(src = "prod1CPU")
process.prod3CPU = testCUDAProducerCPU.clone(src = "prod2CPU")
process.prod4CPU = testCUDAProducerCPU.clone(src = "prod1CPU")
process.prod5CPU = testCUDAProducerCPU.clone()

# Module to decide whether the chain of CUDA modules are run, and to disable a Path in case we don't run on CUDA
from HeterogeneousCore.CUDACore.cudaDeviceChooserFilter_cfi import cudaDeviceChooserFilter
process.prodCUDADeviceFilter = cudaDeviceChooserFilter.clone()

from HeterogeneousCore.CUDACore.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst
from HeterogeneousCore.CUDACore.testCUDAProducerGPU_cfi import testCUDAProducerGPU
from HeterogeneousCore.CUDACore.testCUDAProducerGPUEW_cfi import testCUDAProducerGPUEW
from HeterogeneousCore.CUDACore.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU

# GPU producers
process.prod1CUDA = testCUDAProducerGPUFirst.clone(src = "prodCUDADeviceFilter")
process.prod2CUDA = testCUDAProducerGPU.clone(src = "prod1CUDA")
process.prod3CUDA = testCUDAProducerGPU.clone(src = "prod2CUDA")
process.prod4CUDA = testCUDAProducerGPUEW.clone(src = "prod1CUDA")
process.prod5CUDA = testCUDAProducerGPUFirst.clone(src = "prodCUDADeviceFilter")

# Modules to copy data from GPU to CPU (as "on demand" as any other
# EDProducer, i.e. according to consumes() and prefetching)
process.prod1FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod1CUDA")
process.prod2FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod2CUDA")
process.prod3FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod3CUDA")
process.prod4FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod4CUDA")
process.prod5FromCUDA = testCUDAProducerGPUtoCPU.clone(src = "prod5CUDA")

# These ones are to provide backwards compatibility to the downstream
# clients. To be replaced with an enhanced version of EDAlias (with an
# ordered fallback mechanism).
from HeterogeneousCore.CUDACore.testCUDAProducerFallback_cfi import testCUDAProducerFallback
process.prod1 = testCUDAProducerFallback.clone(src = ["prod1FromCUDA", "prod1cpu"])
process.prod2 = testCUDAProducerFallback.clone(src = ["prod2FromCUDA", "prod2cpu"])
process.prod3 = testCUDAProducerFallback.clone(src = ["prod3FromCUDA", "prod3cpu"])
process.prod4 = testCUDAProducerFallback.clone(src = ["prod4FromCUDA", "prod4cpu"])
process.prod5 = testCUDAProducerFallback.clone(src = ["prod5FromCUDA", "prod5cpu"])

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3_*_*",
        "keep *_prod4_*_*",
        "keep *_prod5_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)

process.prodCPU1 = cms.Path(
    ~process.prodCUDADeviceFilter +
    process.prod1CPU +
    process.prod2CPU +
    process.prod3CPU +
    process.prod4CPU
)
process.prodCUDA1 = cms.Path(
    process.prodCUDADeviceFilter +
    process.prod1CUDA +
    process.prod2CUDA +
    process.prod3CUDA +
    process.prod4CUDA
)

process.prodCPU5 = cms.Path(
    ~process.prodCUDADeviceFilter +
    process.prod5CPU
)
process.prodCUDA5 = cms.Path(
    process.prodCUDADeviceFilter +
    process.prod5CUDA
)

process.t = cms.Task(
    # Eventually the goal is to specify these as part of a Task,
    # but (at least) as long as the fallback mechanism is implemented
    # with an EDProducer, they must be in a Path.
#    process.prod2CPU, process.prod3CPU, process.prod4CPU,
#    process.prod2CUDA, process.prod3CUDA, process.prod4CUDA,

    process.prod1FromCUDA, process.prod2FromCUDA, process.prod3FromCUDA, process.prod4FromCUDA, process.prod5FromCUDA,
    process.prod1, process.prod2, process.prod3, process.prod4, process.prod5,
)
process.p = cms.Path()
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
