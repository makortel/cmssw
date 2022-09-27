import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ProcessAccelerator.')

parser.add_argument("--enableOther", help="Enable other accelerator", action="store_true")
parser.add_argument("--setOtherInResolver", help="Set the default variant to 'other' in module type resolver", action="store_true")
parser.add_argument("--setCpuInResolver", help="Set the default variant to 'cpu' in module type resolver", action="store_true")
parser.add_argument("--accelerators", type=str, help="Comma-separated string for accelerators to enable")

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

class ProcessAcceleratorTest(cms.ProcessAccelerator):
    def __init__(self):
        super(ProcessAcceleratorTest,self).__init__()
        self._labels = ["other"]
        self._enabled = []
        if args.enableOther:
            self._enabled.append("other")
    def labels(self):
        return self._labels
    def enabledLabels(self):
        return self._enabled
    def moduleTypeResolver(self):
        variant = ""
        if args.setOtherInResolver:
            variant = "other"
        if args.setCpuInResolver:
            variant = "cpu"
        return ("edm::test::ConfigurableTestTypeResolverMakerPlugin", cms.untracked.PSet(
            variant = cms.untracked.string(variant)
        ))

process = cms.Process("PROD1")

process.add_(ProcessAcceleratorTest())

process.source = cms.Source("EmptySource",
    firstRun = cms.untracked.uint32(1),
    firstLuminosityBlock = cms.untracked.uint32(1),
    firstEvent = cms.untracked.uint32(1),
    numberEventsInLuminosityBlock = cms.untracked.uint32(1),
    numberEventsInRun = cms.untracked.uint32(1)
)
process.maxEvents.input = 3
if args.accelerators is not None:
    process.options.accelerators = args.accelerators.split(",")

# EventSetup
process.emptyESSourceA = cms.ESSource("EmptyESSource",
    recordName = cms.string("ESTestRecordA"),
    firstValid = cms.vuint32(1,2,3),
    iovIsRunNotTime = cms.bool(True)
)

process.esTestProducerA = cms.ESProducer("ESTestProducerA", valueCpu = cms.int32(10), valueOther = cms.int32(20))

process.esTestAnalyzerA = cms.EDAnalyzer("ESTestAnalyzerA",
    runsToGetDataFor = cms.vint32(1,2,3),
    expectedValues=cms.untracked.vint32(11,12,13)
)

# Event

process.intProducer = cms.EDProducer("generic::IntProducer", valueCpu = cms.int32(1), valueOther = cms.int32(2))

process.intConsumer = cms.EDAnalyzer("IntTestAnalyzer",
    moduleLabel = cms.untracked.InputTag("intProducer"),
    valueMustMatch = cms.untracked.int32(1)
)

if args.enableOther and ("other" in process.options.accelerators or "*" in process.options.accelerators):
    process.intConsumer.valueMustMatch = 2
    process.esTestAnalyzerA.expectedValues = [21, 22, 23]
    if args.setCpuInResolver:
        process.intConsumer.valueMustMatch = 1
    process.esTestAnalyzerA.expectedValues = [11, 12, 13]
if args.setOtherInResolver:
    process.intConsumer.valueMustMatch = 2
    process.esTestAnalyzerA.expectedValues = [21, 22, 23]

process.t = cms.Task(
    process.intProducer
)
process.p = cms.Path(
    process.intConsumer,
    process.t
)
