import FWCore.ParameterSet.Config as cms
process = cms.Process("Test")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource")

process.plugin1 = cms.Plugin("Dummy1")

process.plugin2 = cms.Plugin("Dummy2",
    fromConfig = cms.bool(False)
)

process.plugin3 = cms.Plugin("Dummy2",
    fromConfig = cms.bool(True),
    value = cms.int32(3)
)

process.test1 = cms.EDAnalyzer("TestPluginFactoryService",
    plugin1 = cms.string("plugin1"),
    plugin2 = cms.string("plugin2"),
    plugin2ExpectedValue = cms.int32(2)
)

process.test2 = cms.EDAnalyzer("TestPluginFactoryService",
    plugin1 = cms.string("plugin1"),
    plugin2 = cms.string("plugin3"),
    plugin2ExpectedValue = cms.int32(3)
)

process.p = cms.Path(process.test1 + process.test2)
