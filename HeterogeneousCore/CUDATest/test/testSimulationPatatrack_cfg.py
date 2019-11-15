import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

import six

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
options.register('simple',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Simple application variant (no default)")
options.register('generic',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Generic application variant (no default)")
options.register('genericOrdered',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Should the generic DAG be run in order (1) or unordered (0, default)")
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
options.register('gangStrategy',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Ganging strategy. Default (empty) means to deduce gangSize from numberOfGangs (or vice versa) and numberOfStreams")
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
options.register('singleModule',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Merge all modules into a single module (default 0)")
options.register('blocking',
                 0,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.int,
                 "Replace ExternalWork with blocking wait (default 0)")
options.register('stallMonitor',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "StallMonitor log file (default \"\" to disable)")
options.register('histoFileName',
                 "",
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.string,
                 "Path to file to store histogram data (default \"\" to disable)")
options.parseArguments()

if options.variant not in [0,1,2,3,4,5,6,7,99]:
    raise Exception("Incorrect variant value %d, can be 1,2,3,4,5" % options.variant)
if options.genericOrdered not in [0, 1]:
    raise Exception("genericOrdered should be 0 or 1, got %d" % options.genericOrdered)
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
if options.gangStrategy in ["limitedTaskQueue", "limitedTaskQueueV2", "limitedTaskQueueV3", "limitedTaskQueueV4"]:
    if options.limitedTaskQueue < 1:
        raise Exception("Gang strategy 'limitedTaskQueue' requires the option limitedTaskQueue with value >= 1")
elif options.gangStrategy not in ["", "V2"]:
    raise Exception("Invalid gang strategy '%s'" % options.gangStrategy)
if options.serialTaskQueue not in [0, 1]:
    raise Exception("serialTaskQueue should be 0 or 1, got %d" % options.serialTaskQueue)
if options.gangSize > 0 and options.numberOfGangs > 0:
    raise Exception("One  of 'gangSize' and 'numberOfGangs' can be enabled, now both are")
if options.singleModule not in [0, 1]:
    raise Exception("singleModule should be 0 or 1, got %d" % options.singleModule)
if options.blocking not in [0, 1]:
    raise Exception("blocking should be 0 or 1, got %d" % options.blocking)
if options.gangSize > 0 or options.numberOfGangs > 0 or options.serialTaskQueue == 1 or options.limitedTaskQueue > 0:
    options.gpuExternalWork=1

process = cms.Process("Test")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")
if options.stallMonitor != "":
    process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string(options.stallMonitor)))

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
process.SimOperationsService.maxEvents = options.maxEvents

if options.variant == 2:
    process.SimOperationsService.config = "config_transfer.json"
elif options.variant == 3:
    process.SimOperationsService.config = "config_transfer_convert.json"
elif options.variant == 4:
    process.SimOperationsService.config = "config_cpu.json"
elif options.variant == 5:
    process.SimOperationsService.config = "config_cpu_convert.json"
elif options.variant in [6,7]:
    process.SimOperationsService.config = "config_transfer_convert_mock.json"

if options.gpuExternalWork == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_externalWork.json")
elif options.singleModule == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_externalWorkAll.json")
if options.mean == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_mean.json")
if options.collapse == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_collapse.json")
if options.kernelsInCPUNoMem == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_kernelsInCPU_noMem.json")
elif options.fakeCUDA == 1 or options.fakeCUDANoLock == 1:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_fakeCUDA.json")
    if options.fakeCUDANoLock == 1:
        process.SimOperationsService.fakeUseLocks = False
if len(options.configPostfix) > 0:
    process.SimOperationsService.config = process.SimOperationsService.config.value().replace(".json", "_%s.json"%options.configPostfix)

if options.variant == 0:
    process.SimOperationsService.config = "HeterogeneousCore/CUDATest/test/simpleSimulation.json"
elif options.variant == 99:
    process.SimOperationsService.config = options.generic

if options.gangStrategy in ["", "V2"]:
    if options.gangSize > 0:
        gangNumber = options.numberOfStreams / options.gangSize
        if options.numberOfStreams % options.gangSize != 0:
            raise Exception("numberOfStreams (%d) is not divisible by gang size (%d)" % (options.numberOfStreams, options.gangSize))
        if options.maxEvents % options.gangSize != 0:
            raise Exception("maxEvents (%d) is not divisible by gang size (%d)" % (options.maxEvents, options.gangSize))
        process.SimOperationsService.gangSize = options.gangSize
        process.SimOperationsService.gangNumber = gangNumber
        process.SimOperationsService.gangKernelFactor = options.gangKernelFactor
    elif options.numberOfGangs > 0:
        gangSize = options.numberOfStreams / options.numberOfGangs
        if options.numberOfStreams % options.numberOfGangs != 0:
            raise Exception("numberOfStreams (%d) is not divisible by number of gangs (%d)" % (options.numberOfStreams, options.numberOfGangs))
        if options.maxEvents % gangSize != 0:
            raise Exception("maxEvents (%d) is not divisible by gang size (%d)" % (options.maxEvents, gangSize))
        process.SimOperationsService.gangSize = gangSize
        process.SimOperationsService.gangNumber = options.numberOfGangs
        process.SimOperationsService.gangKernelFactor = options.gangKernelFactor
elif options.gangStrategy in ["limitedTaskQueue", "limitedTaskQueueV2", "limitedTaskQueueV3", "limitedTaskQueueV4"]:
    # Maximum size of a gang
    process.SimOperationsService.gangSize = options.numberOfStreams
    # Maximum number of gangs
    process.SimOperationsService.gangNumber = options.numberOfStreams
    process.SimOperationsService.gangKernelFactor = options.gangKernelFactor

print(process.SimOperationsService.dumpPython())

from HeterogeneousCore.CUDATest.testCUDAProducerSimCPU_cfi import testCUDAProducerSimCPU as _testCUDAProducerSimCPU
from HeterogeneousCore.CUDATest.testCUDAProducerSim_cfi import testCUDAProducerSim as _testCUDAProducerSim
from HeterogeneousCore.CUDATest.testCUDAProducerSimEW_cfi import testCUDAProducerSimEW as _testCUDAProducerSimEW
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGanged_cfi import testCUDAProducerSimEWGanged as _testCUDAProducerSimEWGanged
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGangedV2_cfi import testCUDAProducerSimEWGangedV2 as _testCUDAProducerSimEWGangedV2
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGangedLimitedTaskQueue_cfi import testCUDAProducerSimEWGangedLimitedTaskQueue as _testCUDAProducerSimEWGangedLimitedTaskQueue
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGangedLimitedTaskQueueV2_cfi import testCUDAProducerSimEWGangedLimitedTaskQueueV2 as _testCUDAProducerSimEWGangedLimitedTaskQueueV2
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGangedLimitedTaskQueueV3_cfi import testCUDAProducerSimEWGangedLimitedTaskQueueV3 as _testCUDAProducerSimEWGangedLimitedTaskQueueV3
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGangedLimitedTaskQueueV4_cfi import testCUDAProducerSimEWGangedLimitedTaskQueueV4 as _testCUDAProducerSimEWGangedLimitedTaskQueueV4
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWSerialTaskQueue_cfi import testCUDAProducerSimEWSerialTaskQueue as _testCUDAProducerSimEWSerialTaskQueue
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWLimitedTaskQueue_cfi import testCUDAProducerSimEWLimitedTaskQueue as _testCUDAProducerSimEWLimitedTaskQueue
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWSingle_cfi import testCUDAProducerSimEWSingle as _testCUDAProducerSimEWSingle
from HeterogeneousCore.CUDATest.testCUDAProducerSimBlocking_cfi import testCUDAProducerSimBlocking as _testCUDAProducerSimBlocking

testCUDAProducerSimCPU = _testCUDAProducerSimCPU.clone()
testCUDAProducerSim = _testCUDAProducerSim.clone()
testCUDAProducerSimEW = _testCUDAProducerSimEW.clone()
if options.gpuExternalWork == 1:
    testCUDAProducerSim = _testCUDAProducerSimEW.clone()
if options.blocking == 1:
    testCUDAProducerSimEW = _testCUDAProducerSimBlocking
elif options.gangSize > 0 or options.numberOfGangs > 0 or options.gangStrategy != "":
    _gangModule = _testCUDAProducerSimEWGanged.clone()
    if options.gangStrategy == "V2":
        _gangModule = _testCUDAProducerSimEWGangedV2.clone()
    if options.gangStrategy == "limitedTaskQueue":
        _gangModule = _testCUDAProducerSimEWGangedLimitedTaskQueue.clone(
            limit=options.limitedTaskQueue,
            histoOutput=options.histoFileName,
        )
    elif options.gangStrategy == "limitedTaskQueueV2":
        _gangModule = _testCUDAProducerSimEWGangedLimitedTaskQueueV2.clone(
            limit=options.limitedTaskQueue,
            histoOutput=options.histoFileName,
        )
    elif options.gangStrategy == "limitedTaskQueueV3":
        _gangModule = _testCUDAProducerSimEWGangedLimitedTaskQueueV3.clone(
            limit=options.limitedTaskQueue,
            histoOutput=options.histoFileName,
        )
    elif options.gangStrategy == "limitedTaskQueueV4":
        _gangModule = _testCUDAProducerSimEWGangedLimitedTaskQueueV4.clone(
            limit=options.limitedTaskQueue,
            histoOutput=options.histoFileName,
        )
    testCUDAProducerSim = _gangModule.clone()
    testCUDAProducerSimEW = _gangModule.clone()
elif options.serialTaskQueue == 1:
    testCUDAProducerSim = _testCUDAProducerSimEWSerialTaskQueue.clone()
    testCUDAProducerSimEW = _testCUDAProducerSimEWSerialTaskQueue.clone()
elif options.limitedTaskQueue > 0:
    testCUDAProducerSim = _testCUDAProducerSimEWLimitedTaskQueue.clone(limit=options.limitedTaskQueue)
    testCUDAProducerSimEW = _testCUDAProducerSimEWLimitedTaskQueue.clone(limit=options.limitedTaskQueue)

# Module declarations
if options.singleModule:
    process.theModule = _testCUDAProducerSimEWSingle.clone()
if options.variant in [1,2,3,6,7]:
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

    process.p = cms.Path(
        process.offlineBeamSpot + 
        process.offlineBeamSpotCUDA + 
        process.siPixelClustersCUDAPreSplitting + 
        process.siPixelRecHitsCUDAPreSplitting + 
        process.caHitNtupletCUDA + 
        process.pixelVertexCUDA
    )
    if options.singleModule:
        process.theModule.modules.extend([
            "offlineBeamSpot",
            "offlineBeamSpotCUDA",
            "siPixelClustersCUDAPreSplitting",
            "siPixelRecHitsCUDAPreSplitting",
            "caHitNtupletCUDA",
            "pixelVertexCUDA"
        ])
        process.p = cms.Path(process.theModule)

    if options.variant in [2,3,6,7]:
        process.pixelTrackSoA = testCUDAProducerSimEW.clone(
            cudaSrcs = ["caHitNtupletCUDA"],
            produce=True
        )
        process.pixelVertexSoA = testCUDAProducerSimEW.clone(
            cudaSrcs = ["pixelVertexCUDA"],
            produce=True
        )
        process.p += (process.pixelTrackSoA+process.pixelVertexSoA)
        if options.singleModule:
            process.theModule.modules.extend([
                "pixelTrackSoA",
                "pixelVertexSoA"
            ])
            process.p = cms.Path(process.theModule)

        if options.variant in [3,6,7]:
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
            if options.singleModule:
                process.theModule.modules.extend([
                    "siPixelDigisSoA",
                    "siPixelDigisClustersPreSplitting",
                    "siPixelRecHitsLegacyPreSplitting",
                    "pixelTracks",
                    "pixelVertices",
                ])
                process.p = cms.Path(process.theModule)
                del process.outPath

            if options.variant in [6,7]:
                process.mockIndependent = testCUDAProducerSimCPU.clone(
                    produce = True
                )
                process.t.add(process.mockIndependent)
                process.out.outputCommands.append("keep *_mockIndependent_*_*")
                if options.singleModule:
                    process.theModule.modules.append("mockIndependent")
                if options.variant == 7:
                    process.mockIndependent.srcs = ["pixelTracks", "pixelVertices"]

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
    process.p = cms.Path(
        process.offlineBeamSpot + 
        process.siPixelDigis + 
        process.siPixelClustersPreSplitting +
        process.siPixelRecHitHostSoA + 
        process.pixelTrackSoA + 
        process.pixelVertexSoA
    )
    if options.singleModule:
        process.theModule.modules.extend([
            "offlineBeamSpot",
            "siPixelDigis",
            "siPixelClustersPreSplitting",
            "siPixelRecHitHostSoA",
            "pixelTrackSoA",
            "pixelVertexSoA"
        ])
        process.p = cms.Path(process.theModule)

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
elif options.variant == 99:
    import json
    with open(options.generic) as f:
        config = json.load(f)
    for modName, modType in six.iteritems(config["moduleDeclarations"]):
        setattr(process, modName, {
            "SimCPU": testCUDAProducerSimCPU,
            "Sim": testCUDAProducerSim,
            "SimEW": testCUDAProducerSimEW
            }[modType].clone(produce=True))
    for modName, inputs in six.iteritems(config["moduleConsumes"]):
        if modName == "_out":
            continue
        getattr(process, modName).srcs = [str(x) for x in inputs]
    if options.genericOrdered == 1:
        process.s = cms.Sequence()
        for modName in config["moduleSequence"]:
            process.s += getattr(process, modName)
        process.p = cms.Path(process.s)
    else:
        process.t = cms.Task()
        for modName in config["moduleSequence"]:
            process.t.add(getattr(process, modName))
        process.p = cms.Path(process.t)
        process.out = cms.OutputModule("AsciiOutputModule",
            outputCommands = cms.untracked.vstring(
                ["keep *_%s_*_*" % str(x) for x in config["moduleConsumes"]["_out"]]
            ),
            verbosity = cms.untracked.uint32(0),
        )
        process.outPath = cms.EndPath(process.out)
elif options.variant == 0:
    sv = options.simple
    def finalize(*modnames):
        process.t = cms.Task(*[getattr(process, m) for m in modnames])
        process.p = cms.Path(process.t)
        process.out = cms.OutputModule("AsciiOutputModule",
            outputCommands = cms.untracked.vstring(["keep *_%s_*_*" % m for m in modnames]),
            verbosity = cms.untracked.uint32(0),
        )
        process.outPath = cms.EndPath(process.out)

    if sv == "cpu1s":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True)
        finalize("cpu1s1")
    elif sv == "cpu1s_cpu1s_par":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True)
        process.cpu1s2 = testCUDAProducerSimCPU.clone(produce=True)
        finalize("cpu1s1", "cpu1s2")
    elif sv == "gpu1s_cpu1s_par":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True)
        process.gpu1s1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu1s1")
    elif sv == "gpu1s_cpu2s_par":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True)
        process.gpu2s1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu2s1")
    elif sv == "gpu500ms_cpu1s_par":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True)
        process.gpu500ms1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu500ms1")
    elif sv == "cpu1s_cpu1s_seq":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True, srcs=["cpu1s2"])
        process.cpu1s2 = testCUDAProducerSimCPU.clone(produce=True)
        finalize("cpu1s1", "cpu1s2")
    elif sv == "gpu1s_cpu1s_seq":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True, srcs=["gpu1s1"])
        process.gpu1s1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu1s1")
    elif sv == "gpu2s_cpu1s_seq":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True, srcs=["gpu2s1"])
        process.gpu2s1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu2s1")
    elif sv == "gpu500ms_cpu1s_seq":
        process.cpu1s1 = testCUDAProducerSimCPU.clone(produce=True, srcs=["gpu500ms1"])
        process.gpu500ms1 = testCUDAProducerSimEW.clone(produce=True)
        finalize("cpu1s1", "gpu500ms1")
    else:
        raise Exception("Invalid value for simple '%s'" % sv)
else:
    raise Exception("Unknown variant %d" % options.variant)

#process.t = cms.Task(process.offlineBeamSpot, process.offlineBeamSpotCUDA, process.siPixelClustersCUDAPreSplitting, process.siPixelRecHitsCUDAPreSplitting, process.caHitNtupletCUDA, process.pixelVertexCUDA)
#process.p = cms.Path(process.t)

#process.p = cms.Path(process.offlineBeamSpot)
#process.maxEvents.input = 10

#process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.load('HeterogeneousCore.CUDAServices.NVProfilerService_cfi')
#process.Tracer = cms.Service("Tracer")
#process.options.wantSummary = False

#print process.dumpPython()
