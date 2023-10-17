import FWCore.ParameterSet.Config as cms

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ConditionalTasks.')

parser.add_argument("--filterSucceeds", help="Have filter succeed", action="store_true")
parser.add_argument("--reverseDependencies", help="Switch the order of dependencies", action="store_true")
parser.add_argument("--testAlias", help="Get data from an alias", action="store_true")
parser.add_argument("--testView", help="Get data via a view", action="store_true")
parser.add_argument("--aliasWithStar", help="when using testAlias use '*' as type", action="store_true")

args = parser.parse_args()

process = cms.Process("Test")

process.source = cms.Source("EmptySource")

process.maxEvents.input = 1

process.a = cms.EDProducer("IntProducer", ivalue = cms.int32(1))
process.b = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("a")))

process.f1 = cms.EDFilter("IntProductFilter", label = cms.InputTag("b"))

process.c = cms.EDProducer("IntProducer", ivalue = cms.int32(2))
process.d = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("c")))
process.e = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("d")))

process.nonConsumedEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.nonPathConsumedEvent = cms.EDProducer("edmtest::TestModuleDeleteProducer")
process.nonPathConsumedProcess = cms.EDProducer("edmtest::TestModuleDeleteInProcessProducer")
process.nonPathConsumedLumi = cms.EDProducer("edmtest::TestModuleDeleteInLumiProducer",
    srcBeginProcess = cms.untracked.VInputTag("nonPathConsumedProcess")
)
process.nonPathConsumedRun = cms.EDProducer("edmtest::TestModuleDeleteInRunProducer",
    srcBeginLumi = cms.untracked.VInputTag("nonPathConsumedLumi")
)
process.nonPathConsumedConsumer = cms.EDProducer("edmtest::TestModuleDeleteProducer",
    srcEvent = cms.untracked.VInputTag("nonPathConsumedEvent"),
    srcBeginRun = cms.untracked.VInputTag("nonPathConsumedRun"),
)
process.nonConsumedConditionalTask = cms.ConditionalTask(
    process.nonConsumedEvent,
    process.nonPathConsumedEvent,
    process.nonPathConsumedProcess,
    process.nonPathConsumedLumi,
    process.nonPathConsumedRun,
    process.nonPathConsumedConsumer
)

process.prodOnPath = cms.EDProducer("AddIntsProducer", labels = cms.VInputTag(cms.InputTag("d"), cms.InputTag("e")))

if args.filterSucceeds:
    threshold = 1
else:
    threshold = 3

process.f2 = cms.EDFilter("IntProductFilter", label = cms.InputTag("e"), threshold = cms.int32(threshold))

if args.reverseDependencies:
    process.d.labels[0]=cms.InputTag("e")
    process.e.labels[0]=cms.InputTag("c")
    process.f2.label = cms.InputTag("d")

if args.testView:
  process.f3 = cms.EDAnalyzer("SimpleViewAnalyzer",
    label = cms.untracked.InputTag("f"),
    sizeMustMatch = cms.untracked.uint32(10),
    checkSize = cms.untracked.bool(False)
  )
  process.f = cms.EDProducer("OVSimpleProducer", size = cms.int32(10))
  producttype = "edmtestSimplesOwned"
else:
  process.f= cms.EDProducer("IntProducer", ivalue = cms.int32(3))
  process.f3 = cms.EDFilter("IntProductFilter", label = cms.InputTag("f"))
  producttype = "edmtestIntProduct"

if args.testAlias:
  if args.aliasWithStar:
    producttype = "*"

  process.f3.label = "aliasToF"
  process.aliasToF = cms.EDAlias(
      f = cms.VPSet(
          cms.PSet(
              type = cms.string(producttype),
          )
      )
  )


process.p = cms.Path(process.f1+process.prodOnPath+process.f2+process.f3, cms.ConditionalTask(process.a, process.b, process.c, process.d, process.e, process.f, process.nonConsumedConditionalTask))

process.tst = cms.EDAnalyzer("IntTestAnalyzer", moduleLabel = cms.untracked.InputTag("f"), valueMustMatch = cms.untracked.int32(3), 
                                                       valueMustBeMissing = cms.untracked.bool(not args.filterSucceeds))
process.nonConsumedNonPathConsumer = cms.EDAnalyzer("edmtest::GenericIntsAnalyzer",
    inputShouldBeMissing = cms.untracked.bool(True),
    srcEvent = cms.untracked.VInputTag("nonPathConsumedConsumer")
)
process.intAnalyzerDelete = cms.EDAnalyzer("edmtest::TestModuleDeleteAnalyzer")

process.endp = cms.EndPath(process.tst+process.intAnalyzerDelete+process.nonConsumedNonPathConsumer)

#process.add_(cms.Service("Tracer"))
#process.options.wantSummary=True
process.MessageLogger.DeleteModules = dict()
