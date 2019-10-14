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
options.register('mean',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use configuration with mean operation times (default 0)")
options.register('collapse',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use configuration with operations collapsed within a module (default 0)")
options.register('kernelsInCPUNoMem',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use configuration with kernels ran in CPU and without memory operations (default 0)")
options.register('fakeCUDA',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use configuration with CUDA operations faked in CPU (default 0)")
options.register('fakeCUDANoLock',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use configuration with CUDA operations faked in CPU without locks (default 0)")
options.register('gangSize',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Size of a gang, number of gangs is calculated automatically. -1 to not enable ganging. Conflicts with 'numberOfGangs'. Value > 0 0 implies gpuExternalWork=1")
options.register('numberOfGangs',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Number of gangs, size of a gang is calculated automatically. -1 to not enagle ganging. Conflicts with 'gangSize'. Value > 0 0 implies gpuExternalWork=1")
options.register('gangKernelFactor',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.float,
                 "Kernel time factor. Should be between 0 and 1.")
options.register('serialTaskQueue',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Use SerialTaskQueue. Value 1 implies gpuExternalWork=1.")
options.register('limitedTaskQueue',
                 -1,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Limit in LimitedTaskQueue. Value <= 0 means disabled. Value > 0 implies gpuExternalWork=1")
options.register('configPostfix',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Generic postfix for configuration JSON file")

options.parseArguments()

if options.variant not in [1,2,3,4,5]:
    raise Exception("Incorrect variant value %d, can be 1,2,3,4,5" % options.variant)
if options.gpuExternalWork not in [0, 1]:
    raise Exception("gpuExternalWork should be 0 or 1, got %d" % options.gpuExternalWork)
if options.mean not in [0, 1]:
    raise Exception("mean should be 0 or 1, got %d" % options.mean)
if options.collapse not in [0, 1]:
    raise Exception("collapse should be 0 or 1, got %d" % options.collapse)
if options.kernelsInCPUNoMem not in [0, 1]:
    raise Exception("kernelsInCPUNoMem should be 0 or 1, got %d" % options.kernelsInCPUNoMem)
if options.fakeCUDA not in [0, 1]:
    raise Exception("fakeCUDA should be 0 or 1, got %d" % options.fakeCUDA)
if options.fakeCUDANoLock not in [0, 1]:
    raise Exception("fakeCUDANoLock should be 0 or 1, got %d" % options.fakeCUDANoLock)
if options.serialTaskQueue not in [0, 1]:
    raise Exception("serialTaskQueue should be 0 or 1, got %d" % options.serialTaskQueue)
if options.gangSize > 0 and options.numberOfGangs > 0:
    raise Exception("One  of 'gangSize' and 'numberOfGangs' can be enabled, now both are")
if options.gangSize > 0 or options.numberOfGangs > 0 or options.serialTaskQueue == 1 or options.limitedTaskQueue > 0:
    options.gpuExternalWork=1

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
process.SimOperationsService.cpuCalibration = "HeterogeneousCore/CUDATest/test/cpuCalibration.json"
process.SimOperationsService.cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json"

if options.variant == 2:
    process.SimOperationsService.config = "config_transfer.json"
elif options.variant == 3:
    process.SimOperationsService.config = "config_transfer_convert.json"
elif options.variant == 4:
    process.SimOperationsService.config = "config_cpu.json"
elif options.variant == 5:
    process.SimOperationsService.config = "config_cpu_convert.json"

if options.gpuExternalWork == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_externalWork.json")
elif options.mean == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_mean.json")
elif options.collapse == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_collapse.json")
elif options.kernelsInCPUNoMem == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_kernelsInCPU_noMem.json")
elif options.fakeCUDA == 1 or options.fakeCUDANoLock == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_fakeCUDA.json")
    if options.fakeCUDANoLock == 1:
        process.SimOperationsService.fakeUseLocks = False
if len(options.configPostfix) > 0:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_%s.json"%options.configPostfix)

if options.gangSize > 0:
    gangNumber = options.numberOfStreams / options.gangSize
    if options.numberOfStreams % options.gangSize != 0:
        raise Exception("numberOfStreams (%d) is not divisible by gang size (%d)" % (options.numberOfStreams, options.gangSize))
    process.SimOperationsService.gangSize = options.gangSize
    process.SimOperationsService.gangNumber = gangNumber
    process.SimOperationsService.gangKernelFactor = options.gangKernelFactor
elif options.numberOfGangs > 0:
    gangSize = options.numberOfStreams / options.numberOfGangs
    if options.numberOfStreams % options.numberOfGangs != 0:
        raise Exception("numberOfStreams (%d) is not divisible by number of gangs (%d)" % (options.numberOfStreams, options.numberOfGangs))
    process.SimOperationsService.gangSize = gangSize
    process.SimOperationsService.gangNumber = options.numberOfGangs
    process.SimOperationsService.gangKernelFactor = options.gangKernelFactor

from HeterogeneousCore.CUDATest.testCUDAProducerSimCPU_cfi import testCUDAProducerSimCPU as _testCUDAProducerSimCPU
from HeterogeneousCore.CUDATest.testCUDAProducerSim_cfi import testCUDAProducerSim as _testCUDAProducerSim
from HeterogeneousCore.CUDATest.testCUDAProducerSimEW_cfi import testCUDAProducerSimEW as _testCUDAProducerSimEW
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGanged_cfi import testCUDAProducerSimEWGanged as _testCUDAProducerSimEWGanged
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWSerialTaskQueue_cfi import testCUDAProducerSimEWSerialTaskQueue as _testCUDAProducerSimEWSerialTaskQueue
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWLimitedTaskQueue_cfi import testCUDAProducerSimEWLimitedTaskQueue as _testCUDAProducerSimEWLimitedTaskQueue

testCUDAProducerSimCPU = _testCUDAProducerSimCPU.clone()
testCUDAProducerSim = _testCUDAProducerSim.clone()
testCUDAProducerSimEW = _testCUDAProducerSimEW.clone()
if options.gpuExternalWork == 1:
    testCUDAProducerSim = _testCUDAProducerSimEW.clone()
if options.gangSize > 0:
    testCUDAProducerSim = _testCUDAProducerSimEWGanged.clone()
    testCUDAProducerSimEW = _testCUDAProducerSimEWGanged.clone()
elif options.serialTaskQueue == 1:
    testCUDAProducerSim = _testCUDAProducerSimEWSerialTaskQueue.clone()
    testCUDAProducerSimEW = _testCUDAProducerSimEWSerialTaskQueue.clone()
elif options.limitedTaskQueue > 0:
    testCUDAProducerSim = _testCUDAProducerSimEWLimitedTaskQueue.clone(limit=options.limitedTaskQueue)
    testCUDAProducerSimEW = _testCUDAProducerSimEWLimitedTaskQueue.clone(limit=options.limitedTaskQueue)

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

elif options.variant in [4,5]:
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

    if options.variant == 5:
        process.pixelTracks = testCUDAProducerSimCPU.clone(
            srcs = ["offlineBeamSpot", "siPixelRecHitHostSoA", "pixelTrackSoA"],
            produce=True
        )
        process.pixelVertices = testCUDAProducerSimCPU.clone(
            srcs = ["offlineBeamSpot", "pixelTracks", "pixelVertexSoA"],
            produce=True
        )
        process.t = cms.Task(process.offlineBeamSpot,
                             process.siPixelDigis,
                             process.siPixelClustersPreSplitting,
                             process.siPixelRecHitHostSoA,
                             process.pixelTrackSoA,
                             process.pixelVertexSoA,
                             process.pixelTracks,
                             process.pixelVertices
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

#print process.dumpPython()
