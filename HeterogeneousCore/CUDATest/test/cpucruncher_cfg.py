import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
options.register('maxEvents',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of events")
options.register('numberOfThreads',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of threads.")
options.register('numberOfStreams',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of streams.")
options.parseArguments()

process = cms.Process("Test")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
interval = 1
if options.maxEvents >= 20:
    interval = options.maxEvents/10
process.MessageLogger.cerr.FwkReport.reportEvery = interval

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numberOfThreads),
    numberOfStreams = cms.untracked.uint32(options.numberOfStreams)
)

from HeterogeneousCore.CUDATest.testCUDAProducerSimCPU_cfi import testCUDAProducerSimCPU
process.load("HeterogeneousCore.CUDATest.SimOperationsService_cfi")
process.SimOperationsService.config = "HeterogeneousCore/CUDATest/test/cpucruncher.json"
process.SimOperationsService.cpuCalibration = "HeterogeneousCore/CUDATest/test/cpuCalibration.json"
process.SimOperationsService.cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json"

process.crunch = testCUDAProducerSimCPU.clone()
process.p = cms.Path(process.crunch)
