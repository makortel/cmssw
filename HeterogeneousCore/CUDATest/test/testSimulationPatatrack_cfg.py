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
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))
interval = 1
if options.maxEvents >= 20:
    interval = options.maxEvents/10
process.MessageLogger.cerr.FwkReport.reportEvery = interval

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numberOfThreads),
    numberOfStreams = cms.untracked.uint32(options.numberOfStreams),
    wantSummary = cms.untracked.bool(True)
)

# CUSTOMIZE
#process.maxEvents.input = 4200
#process.MessageLogger.cerr.FwkReport.reportEvery = 100
# END

from HeterogeneousCore.CUDATest.testCUDAProducerSimEW_cfi import testCUDAProducerSimEW as _testCUDAProducerSimEW
from HeterogeneousCore.CUDATest.testCUDAProducerSim_cfi import testCUDAProducerSim as _testCUDAProducerSim
custom = dict(
    config = "config.json",
    cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json",
)
testCUDAProducerSimEW = _testCUDAProducerSimEW.clone(**custom)
testCUDAProducerSim = _testCUDAProducerSim.clone(**custom)
# Module declarations
process.offlineBeamSpot = testCUDAProducerSim.clone(produce=True)
process.offlineBeamSpotCUDA = testCUDAProducerSim.clone(produceCUDA=True)

process.siPixelClustersCUDAPreSplitting = testCUDAProducerSimEW.clone(produceCUDA=True)
process.siPixelRecHitsCUDAPreSplitting = testCUDAProducerSim.clone(produceCUDA=True)
process.caHitNtupletCUDA = testCUDAProducerSim.clone(produceCUDA=True)
process.pixelVertexCUDA = testCUDAProducerSim.clone()

# Insert module dependencies here
process.offlineBeamSpotCUDA.srcs = ["offlineBeamSpot"]
process.siPixelRecHitsCUDAPreSplitting.cudaSrcs = ["offlineBeamSpotCUDA", "siPixelClustersCUDAPreSplitting"]
process.caHitNtupletCUDA.cudaSrcs = ["siPixelRecHitsCUDAPreSplitting"]
process.pixelVertexCUDA.cudaSrcs = ["caHitNtupletCUDA"]

#process.t = cms.Task(process.offlineBeamSpot, process.offlineBeamSpotCUDA, process.siPixelClustersCUDAPreSplitting, process.siPixelRecHitsCUDAPreSplitting, process.caHitNtupletCUDA, process.pixelVertexCUDA)
#process.p = cms.Path(process.t)
process.p = cms.Path(process.offlineBeamSpot+process.offlineBeamSpotCUDA+process.siPixelClustersCUDAPreSplitting+process.siPixelRecHitsCUDAPreSplitting+process.caHitNtupletCUDA+process.pixelVertexCUDA)


#process.p = cms.Path(process.offlineBeamSpot)
#process.maxEvents.input = 10

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.load('HeterogeneousCore.CUDAServices.NVProfilerService_cfi')
