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

from HeterogeneousCore.CUDACore.cudaDeviceChooser_cfi import cudaDeviceChooser
process.testDevice = cudaDeviceChooser.clone()

from HeterogeneousCore.CUDACore.testCUDAProducerCPU_cfi import testCUDAProducerCPU
process.prod1cpu = testCUDAProducerCPU.clone()
process.prod2cpu = testCUDAProducerCPU.clone(src = "prod1cpu")
process.prod3cpu = testCUDAProducerCPU.clone(src = "prod2cpu")
process.prod4cpu = testCUDAProducerCPU.clone(src = "prod1cpu")
process.prod5cpu = testCUDAProducerCPU.clone()

process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_prod3cpu_*_*",
        "keep *_prod4cpu_*_*",
        "keep *_prod5cpu_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)


process.t = cms.Task(
    process.testDevice,
    process.prod1cpu, process.prod2cpu, process.prod3cpu, process.prod4cpu, process.prod5cpu
)
process.p = cms.Path()
process.p.associate(process.t)
process.ep = cms.EndPath(process.out)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
