from builtins import range
import FWCore.ParameterSet.Config as cms
import sys

argv = []
foundpy = False
for a in sys.argv:
    if foundpy:
        argv.append(a)
    if ".py" in a:
        foundpy = True

process = cms.Process("READ")

process.source = cms.Source("DQMRootSource",
                            fileNames = cms.untracked.vstring("file:"+argv[0]),
                            enforceGUIDInFileName = cms.untracked.bool(True))

process.reader = cms.EDAnalyzer("DummyReadDQMStore",
                                 runElements = cms.untracked.VPSet(),
                                 lumiElements = cms.untracked.VPSet() )

process.e = cms.EndPath(process.reader)

process.add_(cms.Service("DQMStore", forceResetOnBeginLumi = cms.untracked.bool(True)))
#process.add_(cms.Service("Tracer"))

