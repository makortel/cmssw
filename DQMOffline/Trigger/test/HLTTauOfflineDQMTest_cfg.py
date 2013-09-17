import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(2000)
        )


process.source = cms.Source("PoolSource",
               fileNames = cms.untracked.vstring(
        "/store/data/Run2012D/TauPlusX/AOD/22Jan2013-v1/20000/000260A0-4286-E211-89DC-0030487F1F23.root",
        "/store/data/Run2012D/TauPlusX/AOD/22Jan2013-v1/20000/0085A19F-9187-E211-9305-0025901D484C.root",
        "/store/data/Run2012D/TauPlusX/AOD/22Jan2013-v1/20000/00DF9FCD-8986-E211-B337-0025901D4936.root",
        # MC, remember to change GT
#         "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00037C53-AAD1-E111-B1BE-003048D45F38.root",
#         "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00050BBE-D5D2-E111-BB65-001E67398534.root",
#         "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00B16DF1-8FD1-E111-ADBB-F04DA23BCE4C.root",
                         )
                            )


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.categories.append("HLTTauDQMOffline")
#process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:com10', '') # for data
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '') # for MC

#process.DQMStore = cms.Service("DQMStore")

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

#Load Tau offline DQM
process.load("DQMOffline.Trigger.HLTTauDQMOffline_cff")

process.dqmEnv.subSystemFolder = "HLTOffline/HLTTAU"

#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/B/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


process.p = cms.Path(process.HLTTauDQMOffline*process.dqmEnv)

process.o = cms.EndPath(process.HLTTauDQMOfflineHarvesting*process.HLTTauDQMOfflineQuality*process.dqmSaver)

process.schedule = cms.Schedule(process.p,process.o)
