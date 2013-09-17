import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTVAL")


process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
#process.MessageLogger.categories.append("HLTTauDQMOffline")
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00037C53-AAD1-E111-B1BE-003048D45F38.root",
        "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00050BBE-D5D2-E111-BB65-001E67398534.root",
        "/store/mc/Summer12_DR53X/DYJetsToLL_M-50_TuneZ2Star_8TeV-madgraph-tarball/AODSIM/PU_S10_START53_V7A-v1/0000/00B16DF1-8FD1-E111-ADBB-F04DA23BCE4C.root",
    )
)

# Set GlobalTag (automatically)
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:mc', '')

#Load DQM Services
process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")


#Reconfigure Environment and saver
#process.dqmEnv.subSystemFolder = cms.untracked.string('HLT/HLTTAU')
#process.DQM.collectorPort = 9091
#process.DQM.collectorHost = cms.untracked.string('pcwiscms10')

process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.workflow = cms.untracked.string('/A/N/C')
process.dqmSaver.forceRunNumber = cms.untracked.int32(123)


#Load the Validation
process.load("HLTriggerOffline.Tau.Validation.HLTTauValidation_cff")

#Load The Post processor
process.load("HLTriggerOffline.Tau.Validation.HLTTauPostValidation_cfi")
process.load("HLTriggerOffline.Tau.Validation.HLTTauQualityTests_cff")


#Define the Paths
process.validation = cms.Path(process.HLTTauVal)

process.postProcess = cms.EndPath(process.HLTTauPostVal+process.hltTauRelvalQualityTests+process.dqmSaver)
#process.postProcess = cms.EndPath(process.dqmSaver)
process.schedule =cms.Schedule(process.validation,process.postProcess)



