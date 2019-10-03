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
options.register('gpuExternalWork',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "All GPU modules are ExternalWorks (default 0)")
options.register('gangSize',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Size of a gang. -1 to disable ganging. Value > 0 0 implies gpuExternalWork=1")
options.register('gangKernelFactor',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Kernel time factor. Should be between 0 and 1.")
options.parseArguments()

if options.variant not in [1,2,3,4]:
    raise Exception("Incorrect variant value %d, can be 1,2,3,4" % options.variant)
if options.gpuExternalWork not in [0, 1]:
    raise Exception("gpuExternalWork should be 0 or 1, got %d" % options.gpuExternalWork)
if options.gangSize > 0:
    options.gpuExternalWork = 1

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

process.load("HeterogeneousCore.CUDATest.SimOperationsService_cfi")
process.SimOperationsService.config = "config.json"
process.SimOperationsService.cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json"

if options.variant == 2:
    process.SimOperationsService.config = "config_transfer.json"
elif options.variant == 3:
    process.SimOperationsService.config = "config_transfer_convert.json"
elif options.variant == 4:
    process.SimOperationsService.config = "config_cpu.json"

if options.gpuExternalWork == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.replace(".json", "_ew.json")
if options.gangSize > 0:
    gangNumber = options.numberOfStreams / options.gangSize
    if options.numberOfStreams % options.gangSize != 0:
        raise Exception("numberOfStreams (%d) is not divisible by gang size (%d)" % (options.numberOfStreams, options.gangSize))
    process.SimOperationsService.gangSize = options.gangSize,
    process.SimOperationsService.gangNumber = gangNumber,
    process.SimOperationsService.gangKernelFactor = options.gangKernelFactor

from HeterogeneousCore.CUDATest.testCUDAProducerSimCPU_cfi import testCUDAProducerSimCPU as _testCUDAProducerSimCPU
from HeterogeneousCore.CUDATest.testCUDAProducerSim_cfi import testCUDAProducerSim as _testCUDAProducerSim
from HeterogeneousCore.CUDATest.testCUDAProducerSimEW_cfi import testCUDAProducerSimEW as _testCUDAProducerSimEW
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGanged_cfi import testCUDAProducerSimEWGanged as _testCUDAProducerSimEWGanged

testCUDAProducerSimCPU = _testCUDAProducerSimCPU.clone()
testCUDAProducerSim = _testCUDAProducerSim.clone()
testCUDAProducerSimEW = _testCUDAProducerSimEW.clone()
if options.gpuExternalWork == 1:
    testCUDAProducerSim = _testCUDAProducerSimEW.clone()
if options.gangSize > 0:
    testCUDAProducerSim = _testCUDAProducerSimEWGanged.clone()
    testCUDAProducerSimEW = _testCUDAProducerSimEWGanged.clone()


# Module declarations
if options.variant in [1,2,3]:
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
                process.pixelVertexSoA,
                process.siPixelDigisSoA,
                process.siPixelDigisClustersPreSplitting,
                process.siPixelRecHitsLegacyPreSplitting,
                process.pixelTracks,
                process.pixelVertices,
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

elif options.variant == 4:
    process.offlineBeamSpot = testCUDAProducerSimCPU.clone(produce=True)
    process.siPixelDigis = testCUDAProducerSimCPU.clone(produce=True)
    process.siPixelClustersPreSplitting = testCUDAProducerSimCPU.clone(
        srcs = ["siPixelDigis"],
        produce=True
    )
    process.siPixelRecHitHostSoA = testCUDAProducerSimCPU.clone(
        srcs = ["offlineBeamSpot", "siPixelClustersPreSplitting"],
        produce=True
    )
    process.pixelTrackSoA = testCUDAProducerSimCPU.clone(
        srcs = ["siPixelRecHitHostSoA"],
        produce=True
    )
    process.pixelVertexSoA = testCUDAProducerSimCPU.clone(
        srcs = ["pixelTrackSoA"],
        produce=True
    )
    process.p = cms.Path(process.offlineBeamSpot+process.siPixelDigis+process.siPixelClustersPreSplitting+process.siPixelRecHitHostSoA+process.pixelTrackSoA+process.pixelVertexSoA)

#process.t = cms.Task(process.offlineBeamSpot, process.offlineBeamSpotCUDA, process.siPixelClustersCUDAPreSplitting, process.siPixelRecHitsCUDAPreSplitting, process.caHitNtupletCUDA, process.pixelVertexCUDA)
#process.p = cms.Path(process.t)

#process.p = cms.Path(process.offlineBeamSpot)
#process.maxEvents.input = 10

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.load('HeterogeneousCore.CUDAServices.NVProfilerService_cfi')
