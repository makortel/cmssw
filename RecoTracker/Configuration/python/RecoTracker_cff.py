import FWCore.ParameterSet.Config as cms

from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
# Iterative steps (select by era)
from Configuration.StandardSequences.Eras import eras
if eras.phase1Pixel.isChosen():
    print "Phase1 tracking"
    from RecoTracker.IterativeTracking.Phase1PU70_iterativeTk_cff import *
    from RecoTracker.IterativeTracking.Phase1PU70_ElectronSeeds_cff import *
else:
    print "Run2 tracking"
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
