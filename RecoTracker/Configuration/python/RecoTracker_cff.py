import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
# Iterative steps (select by era)
from Configuration.StandardSequences.Eras import eras
def _loadForRun2(process):
    print "Loading Run2 tracking"
    process.load("RecoTracker.IterativeTracking.iterativeTk_cff")
    process.load("RecoTracker.IterativeTracking.ElectronSeeds_cff")

#    print process.siPixelClustersPreSplitting
def _loadForPhase1PU70(process):
    print "Loading Phase1PU70 tracking"
    process.load("RecoTracker.IterativeTracking.Phase1PU70_iterativeTk_cff")
    process.load("RecoTracker.IterativeTracking.Phase1PU70_ElectronSeeds_cff")
# Note: following names need to be unique
modifyRecoTrackConfigurationRecoTrackerForRun2_ = eras.tracking_run2.makeProcessModifier(_loadForRun2)
modifyRecoTrackConfigurationRecoTrackerForPhase1PU70_ = eras.tracking_phase1PU70.makeProcessModifier(_loadForPhase1PU70)

import copy

#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_cff import *

#BeamHalo tracking
from RecoTracker.Configuration.RecoTrackerBHM_cff import *


#special sequences, such as pixel-less
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *

ckftracks_woBH = cms.Sequence(
    cms.SequencePlaceholder("iterTracking") +
    cms.SequencePlaceholder("electronSeedsSeq") + 
    doAlldEdXEstimators
)
ckftracks = ckftracks_woBH.copy() #+ beamhaloTracksSeq) # temporarily out, takes too much resources

ckftracks_wodEdX = ckftracks.copy()
ckftracks_wodEdX.remove(doAlldEdXEstimators)


ckftracks_plus_pixelless = cms.Sequence(ckftracks*ctfTracksPixelLess)


from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackingGlobalReco = cms.Sequence(ckftracks*trackExtrapolator)
