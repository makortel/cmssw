import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
# Iterative steps
from RecoTracker.IterativeTracking.iterativeTk_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff import *


import copy

#dEdX reconstruction
from RecoTracker.DeDx.dedxEstimators_cff import *

#BeamHalo tracking
from RecoTracker.Configuration.RecoTrackerBHM_cff import *


#special sequences, such as pixel-less
from RecoTracker.Configuration.RecoTrackerNotStandard_cff import *

ckftracks_woBH = cms.Sequence(iterTracking*electronSeedsSeq*doAlldEdXEstimators)
ckftracks = ckftracks_woBH.copy() #+ beamhaloTracksSeq) # temporarily out, takes too much resources

ckftracks_wodEdX = ckftracks.copy()
ckftracks_wodEdX.remove(doAlldEdXEstimators)


ckftracks_plus_pixelless = cms.Sequence(ckftracks*ctfTracksPixelLess)


from RecoJets.JetAssociationProducers.trackExtrapolator_cfi import *
trackingGlobalReco = cms.Sequence(ckftracks*trackExtrapolator)

from Configuration.StandardSequences.Eras import eras
def _modifyForPhase1(process):
    # Need to clear this sequencs too because of the load below
    # It would be nice if cms.Sequence would have .clear() or similar
    electronSeedsSeq.remove(initialStepSeedClusterMask)
    electronSeedsSeq.remove(pixelPairStepSeedClusterMask)
    electronSeedsSeq.remove(mixedTripletStepSeedClusterMask)
    electronSeedsSeq.remove(pixelLessStepSeedClusterMask)
    electronSeedsSeq.remove(tripletElectronSeedLayers)
    electronSeedsSeq.remove(tripletElectronSeeds)
    electronSeedsSeq.remove(tripletElectronClusterMask)
    electronSeedsSeq.remove(pixelPairElectronSeedLayers)
    electronSeedsSeq.remove(pixelPairElectronSeeds)
    electronSeedsSeq.remove(stripPairElectronSeedLayers)
    electronSeedsSeq.remove(stripPairElectronSeeds)
    electronSeedsSeq.remove(newCombinedSeeds)

    process.load("RecoTracker.IterativeTracking.Phase1PU70_ElectronSeeds_cff") # too different from Run2 that patching with era makes no sense
    ckftracks_woBH.replace(electronSeedsSeq, process.electronSeedsSeq)
    ckftracks.replace(electronSeedsSeq, process.electronSeedsSeq)
    ckftracks_wodEdX.replace(electronSeedsSeq, process.electronSeedsSeq)

modifyRecoTrackerConfigurationRecoTrackerForPhase1Pixel_ = eras.phase1Pixel.makeProcessModifier(_modifyForPhase1)
