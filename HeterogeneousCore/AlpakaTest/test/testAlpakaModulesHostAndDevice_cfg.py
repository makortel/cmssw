import FWCore.ParameterSet.Config as cms
import sys
import argparse

# This configuration demonstrates how to run an EDProducer on two
# possibly different backends: one is the "portable" and another is
# explicitly a host backend, and how to handle (one model of)
# ESProducer in such case.

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test various Alpaka module types')

parser.add_argument("--cuda", help="Use CUDA backend", action="store_true")
parser.add_argument("--run", type=int, help="Run number (default: 1)", default=1)

argv = sys.argv[:]
if '--' in argv:
    argv.remove("--")
args, unknown = parser.parse_known_args(argv)

# TODO: just a temporary mechanism until we get something better that
# works also for ES modules. Absolutely NOT for wider use.
def setToCUDA(m):
    m._TypedParameterizable__type = m._TypedParameterizable__type.replace("alpaka_serial_sync", "alpaka_cuda_async")

process = cms.Process('TEST')

process.source = cms.Source('EmptySource',
    firstRun = cms.untracked.uint32(args.run)
)

process.maxEvents.input = 10

process.load('Configuration.StandardSequences.Accelerators_cff')
process.AlpakaServiceSerialSync = cms.Service('AlpakaServiceSerialSync')
if args.cuda:
    process.AlpakaServiceCudaAsync = cms.Service('AlpakaServiceCudaAsync')

process.alpakaESRecordESource = cms.ESSource("EmptyESSource",
    recordName = cms.string('AlpakaESTestRecordE'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

process.esProducerE = cms.ESProducer("cms::alpakatest::TestESProducerE", value = cms.int32(42))

process.alpakaESProducerEHost = cms.ESProducer("TestAlpakaESProducerE")
process.alpakaESProducerEDevice = cms.ESProducer("alpaka_serial_sync::TestAlpakaESTransferE")
if args.cuda:
    setToCUDA(process.alpakaESProducerEDevice)


process.producer = cms.EDProducer("alpaka_serial_sync::TestAlpakaGlobalProducerConsumeE",
    xvalue = cms.PSet(
        alpaka_serial_sync = cms.double(1.0),
        alpaka_cuda_async = cms.double(2.0)
    )
)
if args.cuda:
    setToCUDA(process.producer)
process.producerHost = cms.EDProducer("alpaka_serial_sync::TestAlpakaGlobalProducerConsumeE",
    xvalue = cms.PSet(
        alpaka_serial_sync = cms.double(1.0),
        alpaka_cuda_async = cms.double(2.0)
    )
)

process.compare = cms.EDAnalyzer("TestAlpakaHostDeviceCompare",
    srcHost = cms.untracked.InputTag("producerHost"),
    srcDevice = cms.untracked.InputTag("producer"),
    expectedXdiff = cms.untracked.double(0.0)
)
if args.cuda:
    process.compare.expectedXdiff = -1.0

process.t = cms.Task(process.producer, process.producerHost)
process.p = cms.Path(process.compare, process.t)

#process.add_(cms.Service("Tracer"))
