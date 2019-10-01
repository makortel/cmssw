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
options.register('variant',
                 1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Application variant (default 1)")
options.parseArguments()

if options.variant not in [1,2,3]:
    raise Exception("Incorrect variant value %d, can be 1,2,3" % options.variant)

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
from HeterogeneousCore.CUDATest.testCUDAProducerSimCPU_cfi import testCUDAProducerSimCPU as _testCUDAProducerSimCPU
custom = dict(
    config = "config.json",
    cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json",
)
if options.variant == 2:
    custom["config"] = "config_transfer.json"
elif options.variant == 3:
    custom["config"] = "config_transfer_convert.json"

testCUDAProducerSimEW = _testCUDAProducerSimEW.clone(**custom)
testCUDAProducerSim = _testCUDAProducerSim.clone(**custom)
testCUDAProducerSimCPU = _testCUDAProducerSimCPU.clone(config=custom["config"])

# Module declarations
process.offlineBeamSpot = testCUDAProducerSimCPU.clone(produce=True)
process.offlineBeamSpotCUDA = testCUDAProducerSim.clone(
    srcs = ["offlineBeamSpot"],
    produceCUDA=True,
)

process.siPixelClustersCUDAPreSplitting = testCUDAProducerSimEW.clone(produceCUDA=True)
process.siPixelRecHitsCUDAPreSplitting = testCUDAProducerSim.clone(
    cudaSrcs = ["offlineBeamSpotCUDA", "siPixelClustersCUDAPreSplitting"],
    produceCUDA=True
)
process.caHitNtupletCUDA = testCUDAProducerSim.clone(
    cudaSrcs = ["siPixelRecHitsCUDAPreSplitting"],
    produceCUDA=True
)
process.pixelVertexCUDA = testCUDAProducerSim.clone(
    cudaSrcs = ["caHitNtupletCUDA"],
    produceCUDA=True
)

process.p = cms.Path(process.offlineBeamSpot+process.offlineBeamSpotCUDA+process.siPixelClustersCUDAPreSplitting+process.siPixelRecHitsCUDAPreSplitting+process.caHitNtupletCUDA+process.pixelVertexCUDA)

if options.variant in [2,3]:
    process.pixelTrackSoA = testCUDAProducerSimEW.clone(
        cudaSrcs = ["caHitNtupletCUDA"],
        produce=True
    )
    process.pixelVertexSoA = testCUDAProducerSimEW.clone(
        cudaSrcs = ["pixelVertexCUDA"],
        produce=True
    )
    process.p += (process.pixelTrackSoA+process.pixelVertexSoA)

    if options.variant == 3:
        process.siPixelDigisSoA = testCUDAProducerSimEW.clone(
            cudaSrcs = ["siPixelClustersCUDAPreSplitting"],
            produce=True
        )
        process.siPixelDigisClustersPreSplitting = testCUDAProducerSimCPU.clone(
            srcs = ["siPixelDigisSoA"],
            produce = True
        )
        process.siPixelRecHitsLegacyPreSplitting = testCUDAProducerSimEW.clone(
            cudaSrcs = ["siPixelRecHitsCUDAPreSplitting"],
            srcs = ["siPixelDigisClustersPreSplitting"],
            produce=True
        )
        process.pixelTracks = testCUDAProducerSimCPU.clone(
            srcs = ["pixelTrackSoA", "siPixelRecHitsLegacyPreSplitting"],
            produce = True
        )
        process.pixelVertices = testCUDAProducerSimCPU.clone(
            srcs = ["pixelTracks", "pixelVertexSoA"],
            produce = True
        )
        process.t = cms.Task(
            process.offlineBeamSpot,
            process.offlineBeamSpotCUDA,
            process.siPixelClustersCUDAPreSplitting,
            process.siPixelRecHitsCUDAPreSplitting,
            process.caHitNtupletCUDA,
            process.pixelVertexCUDA,
            process.pixelTrackSoA,
            process.pixelVertexSoA
        )
        process.p = cms.Path(process.t)
        process.out = cms.OutputModule("AsciiOutputModule",
            outputCommands = cms.untracked.vstring(
                "keep *_pixelTracks_*_*",
                "keep *_pixelVertices_*_*",
            ),
            verbosity = cms.untracked.uint32(0),
        )
        process.outPath = cms.EndPath(process.out)

#process.t = cms.Task(process.offlineBeamSpot, process.offlineBeamSpotCUDA, process.siPixelClustersCUDAPreSplitting, process.siPixelRecHitsCUDAPreSplitting, process.caHitNtupletCUDA, process.pixelVertexCUDA)
#process.p = cms.Path(process.t)

#process.p = cms.Path(process.offlineBeamSpot)
#process.maxEvents.input = 10

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.load('HeterogeneousCore.CUDAServices.NVProfilerService_cfi')
