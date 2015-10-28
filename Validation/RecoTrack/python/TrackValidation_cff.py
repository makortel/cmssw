import FWCore.ParameterSet.Config as cms

import SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi 
import SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi 
import Validation.RecoTrack.MultiTrackValidator_cfi
from SimTracker.TrackAssociation.LhcParametersDefinerForTP_cfi import *
from SimTracker.TrackAssociation.CosmicParametersDefinerForTP_cfi import *
from Validation.RecoTrack.PostProcessorTracker_cfi import *
import PhysicsTools.RecoAlgos.recoTrackSelector_cfi

from SimTracker.TrackerHitAssociation.clusterTpAssociationProducer_cfi import *

TrackAssociatorByHitsRecoDenom= SimTracker.TrackAssociation.quickTrackAssociatorByHits_cfi.quickTrackAssociatorByHits.clone(
    ComponentName = cms.string('TrackAssociatorByHitsRecoDenom'),  
    )

from CommonTools.RecoAlgos.recoTrackViewRefSelector_cfi import recoTrackViewRefSelector

# Validation iterative steps
cutsRecoTracksZero = recoTrackViewRefSelector.clone()
cutsRecoTracksZero.algorithm=cms.vstring("iter0")

cutsRecoTracksFirst = recoTrackViewRefSelector.clone()
cutsRecoTracksFirst.algorithm=cms.vstring("iter1")

cutsRecoTracksSecond = recoTrackViewRefSelector.clone()
cutsRecoTracksSecond.algorithm=cms.vstring("iter2")

cutsRecoTracksThird = recoTrackViewRefSelector.clone()
cutsRecoTracksThird.algorithm=cms.vstring("iter3")

cutsRecoTracksFourth = recoTrackViewRefSelector.clone()
cutsRecoTracksFourth.algorithm=cms.vstring("iter4")

cutsRecoTracksFifth = recoTrackViewRefSelector.clone()
cutsRecoTracksFifth.algorithm=cms.vstring("iter5")

cutsRecoTracksSixth = recoTrackViewRefSelector.clone()
cutsRecoTracksSixth.algorithm=cms.vstring("iter6")

cutsRecoTracksNinth = recoTrackViewRefSelector.clone()
cutsRecoTracksNinth.algorithm=cms.vstring("iter9")

cutsRecoTracksTenth = recoTrackViewRefSelector.clone()
cutsRecoTracksTenth.algorithm=cms.vstring("iter10")

# high purity
cutsRecoTracksHp = recoTrackViewRefSelector.clone()
cutsRecoTracksHp.quality=cms.vstring("highPurity")

cutsRecoTracksZeroHp = recoTrackViewRefSelector.clone()
cutsRecoTracksZeroHp.algorithm=cms.vstring("iter0")
cutsRecoTracksZeroHp.quality=cms.vstring("highPurity")

cutsRecoTracksFirstHp = recoTrackViewRefSelector.clone()
cutsRecoTracksFirstHp.algorithm=cms.vstring("iter1")
cutsRecoTracksFirstHp.quality=cms.vstring("highPurity")

cutsRecoTracksSecondHp = recoTrackViewRefSelector.clone()
cutsRecoTracksSecondHp.algorithm=cms.vstring("iter2")
cutsRecoTracksSecondHp.quality=cms.vstring("highPurity")

cutsRecoTracksThirdHp = recoTrackViewRefSelector.clone()
cutsRecoTracksThirdHp.algorithm=cms.vstring("iter3")
cutsRecoTracksThirdHp.quality=cms.vstring("highPurity")

cutsRecoTracksFourthHp = recoTrackViewRefSelector.clone()
cutsRecoTracksFourthHp.algorithm=cms.vstring("iter4")
cutsRecoTracksFourthHp.quality=cms.vstring("highPurity")

cutsRecoTracksFifthHp = recoTrackViewRefSelector.clone()
cutsRecoTracksFifthHp.algorithm=cms.vstring("iter5")
cutsRecoTracksFifthHp.quality=cms.vstring("highPurity")

cutsRecoTracksSixthHp = recoTrackViewRefSelector.clone()
cutsRecoTracksSixthHp.algorithm=cms.vstring("iter6")
cutsRecoTracksSixthHp.quality=cms.vstring("highPurity")

cutsRecoTracksNinthHp = recoTrackViewRefSelector.clone()
cutsRecoTracksNinthHp.algorithm=cms.vstring("iter9")
cutsRecoTracksNinthHp.quality=cms.vstring("highPurity")

cutsRecoTracksTenthHp = recoTrackViewRefSelector.clone()
cutsRecoTracksTenthHp.algorithm=cms.vstring("iter10")
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
                                cutsRecoTracksNinth*
                                cutsRecoTracksNinthHp* 
                                cutsRecoTracksTenth*
                                cutsRecoTracksTenthHp )

# selectors go into separate "prevalidation" sequence
tracksValidation = cms.Sequence( tpClusterProducer * trackValidator)
tracksValidationFS = cms.Sequence( trackValidator )

