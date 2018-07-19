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
#   5
#
# CPU producers
from HeterogeneousCore.CUDACore.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod1cpu = testCUDAProducerCPU.clone()
process.prod2cpu = testCUDAProducerCPU.clone(src = "prod1cpu")
process.prod3cpu = testCUDAProducerCPU.clone(src = "prod2cpu")
process.prod4cpu = testCUDAProducerCPU.clone(src = "prod1cpu")
process.prod5cpu = testCUDAProducerCPU.clone()

# Module to decide whether the chain of CUDA modules are run
from HeterogeneousCore.CUDACore.cudaDeviceChooser_cfi import cudaDeviceChooser
process.testDevice = cudaDeviceChooser.clone()

# Filter to disable a Path in case we don't run on CUDA
from HeterogeneousCore.CUDACore.cudaDeviceFilter_cfi import cudaDeviceFilter
process.testDeviceFilter = cudaDeviceFilter.clone(src = "testDevice")

from HeterogeneousCore.CUDACore.testCUDAProducerGPUFirst_cfi import testCUDAProducerGPUFirst
from HeterogeneousCore.CUDACore.testCUDAProducerGPU_cfi import testCUDAProducerGPU
from HeterogeneousCore.CUDACore.testCUDAProducerGPUtoCPU_cfi import testCUDAProducerGPUtoCPU

# GPU producers
process.prod1gpu = testCUDAProducerGPUFirst.clone(src = "testDevice")
process.prod2gpu = testCUDAProducerGPU.clone(src = "prod1gpu")
process.prod3gpu = testCUDAProducerGPU.clone(src = "prod2gpu")
process.prod4gpu = testCUDAProducerGPU.clone(src = "prod1gpu")
process.prod5gpu = testCUDAProducerGPUFirst.clone(src = "testDevice")

# Modules to copy data from GPU to CPU (as "on demand" as any other
# EDProducer, i.e. according to consumes() and prefetching)
process.prod1gpuOnCpu = testCUDAProducerGPUtoCPU.clone(src = "prod1gpu")
process.prod2gpuOnCpu = testCUDAProducerGPUtoCPU.clone(src = "prod2gpu")
process.prod3gpuOnCpu = testCUDAProducerGPUtoCPU.clone(src = "prod3gpu")
process.prod4gpuOnCpu = testCUDAProducerGPUtoCPU.clone(src = "prod4gpu")
process.prod5gpuOnCpu = testCUDAProducerGPUtoCPU.clone(src = "prod5gpu")

# These ones are to provide backwards compatibility to the downstream
# clients. To be replaced with an enhanced version of EDAlias (with an
# ordered fallback mechanism).
from HeterogeneousCore.CUDACore.testCUDAProducerFallback_cfi import testCUDAProducerFallback
process.prod1 = testCUDAProducerFallback.clone(src = ["prod1gpuOnCpu", "prod1cpu"])
process.prod2 = testCUDAProducerFallback.clone(src = ["prod2gpuOnCpu", "prod2cpu"])
process.prod3 = testCUDAProducerFallback.clone(src = ["prod3gpuOnCpu", "prod3cpu"])
process.prod4 = testCUDAProducerFallback.clone(src = ["prod4gpuOnCpu", "prod4cpu"])
process.prod5 = testCUDAProducerFallback.clone(src = ["prod5gpuOnCpu", "prod5cpu"])

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3_*_*",
        "keep *_prod4_*_*",
        "keep *_prod5_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)

process.prodCPU1 = cms.Path(
    ~process.testDeviceFilter +
    process.prod1cpu +
    process.prod2cpu +
    process.prod3cpu +
    process.prod4cpu
)
process.prodCUDA1 = cms.Path(
    process.testDeviceFilter +
    process.prod1gpu +
    process.prod2gpu +
    process.prod3gpu +
    process.prod4gpu
)

process.prodCPU5 = cms.Path(
    ~process.testDeviceFilter +
    process.prod5gpu
)
process.prodCUDA5 = cms.Path(
    process.testDeviceFilter +
    process.prod5gpu
)

process.t = cms.Task(
    process.testDevice,
#    process.prod2cpu, process.prod3cpu, process.prod4cpu,
#    process.prod2gpu, process.prod3gpu, process.prod4gpu,
    process.prod1gpuOnCpu, process.prod2gpuOnCpu, process.prod3gpuOnCpu, process.prod4gpuOnCpu, process.prod5gpuOnCpu,
    process.prod1, process.prod2, process.prod3, process.prod4, process.prod5,
)
process.p = cms.Path()
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
