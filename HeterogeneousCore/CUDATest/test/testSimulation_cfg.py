import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing()
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

interval = 1000*options.numberOfThreads
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(12*interval))
process.MessageLogger.cerr.FwkReport.reportEvery = interval

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(options.numberOfThreads),
    numberOfStreams = cms.untracked.uint32(options.numberOfStreams)
)

#process.overhead = cms.EDProducer("TestCUDAProducerOverhead")
#process.overhead = cms.EDProducer("TestCUDAProducerOverheadEW")
#process.overhead = cms.EDProducer("TestCUDAProducerOverheadEW", lockMutex=cms.bool(True))
#process.p_overhead = cms.Path(process.overhead)

from HeterogeneousCore.CUDATest.testCUDAProducerSimEW_cfi import testCUDAProducerSimEW
from HeterogeneousCore.CUDATest.testCUDAProducerSimEWGanged_cfi import testCUDAProducerSimEWGanged
#from HeterogeneousCore.CUDATest.testCUDAProducerSimEWSerialTaskQueue_cfi import testCUDAProducerSimEWSerialTaskQueue as testCUDAProducerSimEW
template = testCUDAProducerSimEW.clone()
template = testCUDAProducerSimEWGanged.clone(
    gangSize = 1
)
template.gangNumber = process.options.numberOfStreams.value()
if template.gangNumber.value() == 0:
    template.gangNumber = process.options.numberOfThreads.value()
if template.gangNumber.value() % template.gangSize.value() != 0:
    raise Exception("Number of streams %d is not divisible with the gang size %d" % (template.gangNumber.value(), template.gangSize.value()))
template.gangNumber = template.gangNumber.value()/template.gangSize.value()
process.transfer = template.clone(
    config = "HeterogeneousCore/CUDATest/test/testSimulation.json",
    cudaCalibration = "HeterogeneousCore/CUDATest/test/cudaCalibration.json",
    produce = True
)

process.p = cms.Path(process.transfer)

from HeterogeneousCore.CUDATest.testCUDAProducerCPUCrunch_cfi import testCUDAProducerCPUCrunch
process.cpu1 = testCUDAProducerCPUCrunch.clone()
process.cpu2 = testCUDAProducerCPUCrunch.clone(srcs=["cpu1", "transfer"])

process.cpu1.crunchForSeconds = 100e-6
process.cpu2.crunchForSeconds = 200e-6

process.t_transfer = cms.Task(process.transfer, process.cpu1, process .cpu2)
process.p_transfer = cms.Path(process.t_transfer)
process.out = cms.OutputModule("AsciiOutputModule",
    outputCommands = cms.untracked.vstring(
        "keep *_cpu2_*_*",
    ),
    verbosity = cms.untracked.uint32(0),
)
process.p_out = cms.EndPath(process.out)

process.maxEvents.input = process.maxEvents.input.value()/10
process.MessageLogger.cerr.FwkReport.reportEvery = process.MessageLogger.cerr.FwkReport.reportEvery.value()/10
factor = 1
#process.transfer.numberOfElements = factor*process.transfer.numberOfElements.value()
#process.transfer.kernelLoops = factor*process.transfer.kernelLoops.value()
#process.cpu1.crunchForSeconds = factor*process.cpu1.crunchForSeconds.value()
#process.cpu2.crunchForSeconds = factor*process.cpu2.crunchForSeconds.value()

if factor > 1:
    for i in xrange(1, factor):
        m1 = process.cpu1.clone()
        m2 = process.cpu2.clone(srcs=["cpu1c%d"%i, "transfer"])
        setattr(process, "cpu1c%d"%i, m1)
        setattr(process, "cpu2c%d"%i, m2)
        process.t_transfer.add(m1, m2)
        process.out.outputCommands.append("keep *_cpu2c%d_*_*" % i)

process.maxEvents.input = 16
process.options.numberOfStreams = 4
process.transfer.gangSize = 2
process.transfer.gangNumber = 2
#process.Tracer = cms.Service("Tracer")
#process.out.verbosity = 1
#process.options.numberOfThreads = 8
#process.load('HeterogeneousCore.CUDAServices.NVProfilerService_cfi')
#process.NVProfilerService.skipFirstEvent = True
process.MessageLogger.cerr.FwkReport.reportEvery = 1
