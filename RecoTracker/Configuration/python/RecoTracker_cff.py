import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
# Iterative steps (select by era)
from Configuration.StandardSequences.Eras import eras
def _loadForRun2(process):
    # This needs to be a bit convoluted because Run2_2017 ModifierChain includes both run2_common and phase1Pixel Modifiers
    if eras.phase1Pixel.isChosen():
        return
    print "Loading Run2 tracking"
    process.load("RecoTracker.IterativeTracking.iterativeTk_cff")
    process.load("RecoTracker.IterativeTracking.ElectronSeeds_cff")

#    print process.siPixelClustersPreSplitting
def _loadForPhase1(process):
    print "Loading Phase1 tracking"
    process.load("RecoTracker.IterativeTracking.Phase1PU70_iterativeTk_cff")
    process.load("RecoTracker.IterativeTracking.Phase1PU70_ElectronSeeds_cff")
# Note: following names need to be unique
modifyRecoTrackConfigurationRecoTrackerForRun1_ = eras.Run1.makeProcessModifier(_loadForRun2) # we use run2 configuration for run1
modifyRecoTrackConfigurationRecoTrackerForRun2_ = eras.run2_common.makeProcessModifier(_loadForRun2)
modifyRecoTrackConfigurationRecoTrackerForPhase1_ = eras.phase1Pixel.makeProcessModifier(_loadForPhase1)

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
