import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test AsyncService')
parser.add_argument("--exception", help="Include another module that throws exception in event function", action="store_true")
args = parser.parse_args()

process = cms.Process("TEST")

process.maxEvents.input = 4
process.options.numberOfThreads = 2
process.options.numberOfStreams = 2
process.source = cms.Source("EmptySource")

process.add_(cms.Service("AsyncService"))

process.tester = cms.EDProducer("edmtest::AsyncServiceTester")

process.p = cms.Path(process.tester)
if args.exception:
    process.tester.wait = cms.untracked.bool(True)
    process.add_(cms.Service("edmtest::AsyncServiceTesterService"))

    process.firstEventFilter = cms.EDFilter("ModuloEventIDFilter",
        modulo = cms.uint32(1000),
        offset = cms.uint32(1)
    )
    process.fail = cms.EDProducer("FailingProducer")
    process.p2 = cms.Path(process.firstEventFilter+process.fail)

process.add_(cms.Service("ZombieKillerService", secondsBetweenChecks=cms.untracked.uint32(5)))
