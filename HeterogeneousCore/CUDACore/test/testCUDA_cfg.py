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

process.t = cms.Task(process.testDevice)
process.p = cms.Path()
process.p.associate(process.t)

# Example of limiting the number of EDM streams per device
#process.CUDAService.numberOfStreamsPerDevice = 1
