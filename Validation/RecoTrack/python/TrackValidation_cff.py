import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociatorProducers.trackAssociatorByChi2_cfi 
from SimTracker.TrackAssociatorProducers.quickTrackAssociatorByHits_cfi import *
from SimTracker.TrackAssociation.trackingParticleRecoTrackAsssociation_cfi import *
import Validation.RecoTrack.MultiTrackValidator_cfi
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import CommonTools.RecoAlgos.recoTrackRefSelector_cfi

from SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi import *

# Validation iterative steps
cutsRecoTracksZero = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksZero.algorithm=cms.vstring("initialStep")

cutsRecoTracksFirst = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFirst.algorithm=cms.vstring("lowPtTripletStep")

cutsRecoTracksSecond = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSecond.algorithm=cms.vstring("pixelPairStep")

cutsRecoTracksThird = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksThird.algorithm=cms.vstring("detachedTripletStep")

cutsRecoTracksFourth = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFourth.algorithm=cms.vstring("mixedTripletStep")

cutsRecoTracksFifth = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFifth.algorithm=cms.vstring("pixelLessStep")

cutsRecoTracksSixth = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSixth.algorithm=cms.vstring("tobTecStep")

cutsRecoTracksSeventh = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSeventh.algorithm=cms.vstring("jetCoreRegionalStep")

cutsRecoTracksNinth = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksNinth.algorithm=cms.vstring("muonSeededStepInOut")

cutsRecoTracksTenth = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksTenth.algorithm=cms.vstring("muonSeededStepOutIn")

# high purity
cutsRecoTracksHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksZeroHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksZeroHp.algorithm=cms.vstring("initialStep")
cutsRecoTracksZeroHp.quality=cms.vstring("highPurity")

cutsRecoTracksFirstHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFirstHp.algorithm=cms.vstring("lowPtTripletStep")
cutsRecoTracksFirstHp.quality=cms.vstring("highPurity")

cutsRecoTracksSecondHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSecondHp.algorithm=cms.vstring("pixelPairStep")
cutsRecoTracksSecondHp.quality=cms.vstring("highPurity")

cutsRecoTracksThirdHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksThirdHp.algorithm=cms.vstring("detachedTripletStep")
cutsRecoTracksThirdHp.quality=cms.vstring("highPurity")

cutsRecoTracksFourthHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFourthHp.algorithm=cms.vstring("mixedTripletStep")
cutsRecoTracksFourthHp.quality=cms.vstring("highPurity")

cutsRecoTracksFifthHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksFifthHp.algorithm=cms.vstring("pixelLessStep")
cutsRecoTracksFifthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSixthHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSixthHp.algorithm=cms.vstring("tobTecStep")
cutsRecoTracksSixthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSeventhHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksSeventhHp.algorithm=cms.vstring("jetCoreRegionalStep")
cutsRecoTracksSeventhHp.quality=cms.vstring("highPurity")

cutsRecoTracksNinthHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksNinthHp.algorithm=cms.vstring("muonSeededStepInOut")
cutsRecoTracksNinthHp.quality=cms.vstring("highPurity")

cutsRecoTracksTenthHp = CommonTools.RecoAlgos.recoTrackRefSelector_cfi.recoTrackRefSelector.clone()
cutsRecoTracksTenthHp.algorithm=cms.vstring("muonSeededStepOutIn")
cutsRecoTracksTenthHp.quality=cms.vstring("highPurity")

trackValidator= Validation.RecoTrack.MultiTrackValidator_cfi.multiTrackValidator.clone()

trackValidator.label=cms.VInputTag(cms.InputTag("generalTracks"),
                                   cms.InputTag("cutsRecoTracksHp"),
                                   cms.InputTag("cutsRecoTracksZero"),
                                   cms.InputTag("cutsRecoTracksZeroHp"),
                                   cms.InputTag("cutsRecoTracksFirst"),
                                   cms.InputTag("cutsRecoTracksFirstHp"),
                                   cms.InputTag("cutsRecoTracksSecond"),
                                   cms.InputTag("cutsRecoTracksSecondHp"),
                                   cms.InputTag("cutsRecoTracksThird"),
                                   cms.InputTag("cutsRecoTracksThirdHp"),
                                   cms.InputTag("cutsRecoTracksFourth"),
                                   cms.InputTag("cutsRecoTracksFourthHp"),
                                   cms.InputTag("cutsRecoTracksFifth"),
                                   cms.InputTag("cutsRecoTracksFifthHp"),
                                   cms.InputTag("cutsRecoTracksSixth"),
                                   cms.InputTag("cutsRecoTracksSixthHp"),
                                   cms.InputTag("cutsRecoTracksSeventh"),
                                   cms.InputTag("cutsRecoTracksSeventhHp"),
                                   cms.InputTag("cutsRecoTracksNinth"),
                                   cms.InputTag("cutsRecoTracksNinthHp"),
                                   cms.InputTag("cutsRecoTracksTenth"),
                                   cms.InputTag("cutsRecoTracksTenthHp"),
                                   )
trackValidator.skipHistoFit=cms.untracked.bool(True)
trackValidator.useLogPt=cms.untracked.bool(True)
#trackValidator.minpT = cms.double(-1)
#trackValidator.maxpT = cms.double(3)
#trackValidator.nintpT = cms.int32(40)

# the track selectors
tracksValidationSelectors = cms.Sequence( cutsRecoTracksHp*
                                cutsRecoTracksZero*
                                cutsRecoTracksZeroHp*
                                cutsRecoTracksFirst*
                                cutsRecoTracksFirstHp*
                                cutsRecoTracksSecond*
                                cutsRecoTracksSecondHp*
                                cutsRecoTracksThird*
                                cutsRecoTracksThirdHp*
                                cutsRecoTracksFourth*
                                cutsRecoTracksFourthHp*
                                cutsRecoTracksFifth*
                                cutsRecoTracksFifthHp*
                                cutsRecoTracksSixth*
                                cutsRecoTracksSixthHp* 
                                cutsRecoTracksSeventh*
                                cutsRecoTracksSeventhHp* 
                                cutsRecoTracksNinth*
                                cutsRecoTracksNinthHp* 
                                cutsRecoTracksTenth*
                                cutsRecoTracksTenthHp )
tracksValidationTruth = cms.Sequence(
    tpClusterProducer +
    quickTrackAssociatorByHits +
    trackingParticleRecoTrackAsssociation
)

tracksPreValidation = cms.Sequence(
    tracksValidationSelectors +
    tracksValidationTruth
)

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( trackValidator)
tracksValidationFS = cms.Sequence( trackValidator )

