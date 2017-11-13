import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('RECO',eras.Phase2)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

#adding only recolocalreco
process.load('RecoLocalTracker.Configuration.RecoLocalTracker_cff')

# import VectorHitBuilder                                                                                                                                                      
process.load('RecoLocalTracker.Phase2TrackerRecHits.SiPhase2VectorHitBuilder_cfi')


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Input source
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_9_3_2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_93X_upgrade2023_realistic_v2_2023D17PU200-v1/10000/0253F3E0-C4AA-E711-96FF-E0071B73C620.root'),
    fileNames = cms.untracked.vstring('file:step2.root'),
    secondaryFileNames = cms.untracked.vstring(),
    skipEvents = cms.untracked.uint32(0),
#    inputCommands = cms.untracked.vstring( ('drop *_*_*_HLT') )
    #eventsToProcess = cms.untracked.VEventRange('1:952-1:952')
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step3 nevts:10'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.RECOSIMoutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-RECO'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step3_1event.root'),
    outputCommands = cms.untracked.vstring( ('keep *','drop *_*_*_HLT') ),
    splitLevel = cms.untracked.int32(0)
)

# debug
process.MessageLogger = cms.Service("MessageLogger",
                                    destinations = cms.untracked.vstring("debugVHs"),
                                    debugModules = cms.untracked.vstring("*"),
                                    #categories = cms.untracked.vstring("VectorHitBuilder"),
                                    categories = cms.untracked.vstring("VectorHitBuilder","VectorHitBuilderAlgorithm","VectorHitsBuilderValidation"),
                                    debugVHs = cms.untracked.PSet(threshold = cms.untracked.string("DEBUG"),
                                                                       DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
                                                                       VectorHitBuilder = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       VectorHitBuilderAlgorithm = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       VectorHitsBuilderValidation = cms.untracked.PSet(limit = cms.untracked.int32(-1)),
                                                                       )
                                    )

# Analyzer
process.analysis = cms.EDAnalyzer('VectorHitsBuilderValidation',
    src = cms.string("siPhase2Clusters"),
    VH_acc = cms.InputTag("siPhase2VectorHits", "vectorHitsAccepted"),
    VH_rej = cms.InputTag("siPhase2VectorHits", "vectorHitsRejected"),
    CPE = cms.ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"),
    links = cms.InputTag("simSiPixelDigis", "Tracker"),
    trackingParticleSrc = cms.InputTag('mix', 'MergedTrackTruth'),
)
process.TFileService = cms.Service('TFileService',
    fileName = cms.string('file:VHs_validation.root')
)


# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.trackerlocalreco_step  = cms.Path(process.trackerlocalreco+process.siPhase2VectorHits)
process.analysis_step = cms.Path(process.analysis)
process.RECOSIMoutput_step = cms.EndPath(process.RECOSIMoutput)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.trackerlocalreco_step,process.RECOSIMoutput_step, process.analysis_step)

