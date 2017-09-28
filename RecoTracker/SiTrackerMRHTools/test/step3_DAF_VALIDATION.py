## This is an example with QCD data in the release CMSSW_924
# Auto generated configuration file
# using: 
# Revision: 1.19 
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v 
# with command line options: step3 --conditions auto:run2_mc --scenario pp --mc --era Run2_2016 -n 100 --no_exec --eventcontent DQM --datatier DQMIO -s VALIDATION:tracksValidationStandalone --filein filelist:file_reco_qcd80_120.log --secondfilein filelist:file_digi_qcd80_120.log --fileout step3_inDQM.root
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('VALIDATION',eras.Run2_2016)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Validation_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### Track Refitter
process.load("RecoTracker.TrackProducer.TrackRefitters_cff")
process.load("RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cff")
process.TracksDAF.TrajectoryInEvent = True
process.TracksDAF.src = 'TrackRefitter'
process.TracksDAF.TrajAnnealingSaving = False
process.MRHFittingSmoother.EstimateCut = -1
process.MRHFittingSmoother.MinNumberOfHits = 3

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/91X_mcRun2_asymptotic_v3-v1/00000/42495FF1-AA62-E711-A4F5-0CC47A7C3612.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/91X_mcRun2_asymptotic_v3-v1/00000/4A153C7F-AD62-E711-A80F-0025905B860C.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/91X_mcRun2_asymptotic_v3-v1/00000/6E342C79-AD62-E711-8C8B-0025905B8562.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-RECO/91X_mcRun2_asymptotic_v3-v1/00000/8A7DFEFD-AA62-E711-A685-0CC47A7C3636.root'),
    secondaryFileNames = cms.untracked.vstring('/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/00C76678-A462-E711-AC8D-0CC47A4D764A.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/0E1C6477-A462-E711-9931-0025905B855E.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/2AF92A88-A662-E711-9247-0CC47A4C8E5E.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/74EE9477-A462-E711-867A-0025905B8580.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/BE7FD25C-AB62-E711-A19A-0CC47A7C353E.root', 
        '/store/relval/CMSSW_9_2_4/RelValQCD_Pt_80_120_13/GEN-SIM-DIGI-RAW-HLTDEBUG/91X_mcRun2_asymptotic_v3-v1/00000/F00F3187-A662-E711-A6B2-0CC47A78A42C.root')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:100'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.DQMoutput = cms.OutputModule("DQMRootOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('DQMIO'),
        filterName = cms.untracked.string('')
    ),
    fileName = cms.untracked.string('step3_DAF_inDQM.root'),
    outputCommands = process.DQMEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition
## debug(DAFTrackProducerAlgorithm)
#process.MessageLogger = cms.Service("MessageLogger",
#                                    destinations = cms.untracked.vstring("debugTracking_DAF"),
#                                    debugModules = cms.untracked.vstring("*"),
#                                    categories = cms.untracked.vstring("DAFTrackProducerAlgorithm","TrackValidator"),
#                                    debugTracking_DAF = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
#                                                                      DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                      default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
#                                                                      DAFTrackProducerAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                      TrackValidator = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
#                                                                       )
#                                    )


# Other statements
process.mix.playback = True
process.mix.digitizers = cms.PSet()
for a in process.aliases: delattr(process, a)
process.RandomNumberGeneratorService.restoreStateLabel=cms.untracked.string("randomEngineStateProducer")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

#
process.trackValidatorStandalone.dodEdxPlots = False
process.trackValidatorFromPVStandalone.dodEdxPlots = False
process.trackValidatorFromPVAllTPStandalone.dodEdxPlots = False
process.trackValidatorAllTPEfficStandalone.dodEdxPlots = False 
process.trackValidatorStandalone.label.remove("cutsRecoTracksAK4PFJets")
process.trackValidatorBHadronStandalone.label.remove("cutsRecoTracksAK4PFJets")

# Path and EndPath definitions
process.generalTracks = process.TracksDAF.clone()
process.TrackRefitter.src.setProcessName(cms.InputTag.skipCurrentProcess())
process.daf_step = cms.Path(process.MeasurementTrackerEvent*process.TrackRefitter*process.generalTracks)
process.validation_step = cms.EndPath(process.tracksValidationStandalone)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.daf_step,process.validation_step,process.DQMoutput_step)
from PhysicsTools.PatAlgos.tools.helpers import associatePatAlgosToolsTask
associatePatAlgosToolsTask(process)

# customisation of the process.

# Automatic addition of the customisation function from SimGeneral.MixingModule.fullMixCustomize_cff
from SimGeneral.MixingModule.fullMixCustomize_cff import setCrossingFrameOn 

#call to customisation function setCrossingFrameOn imported from SimGeneral.MixingModule.fullMixCustomize_cff
process = setCrossingFrameOn(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion

#process.Timing = cms.Service("Timing")
