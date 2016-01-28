import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *

earlyGeneralTracksSequence = cms.Sequence(
    earlyGeneralTracks
)

from Configuration.StandardSequences.Eras import eras
def _modifyForPhase1(process):
    # Need to clear because of the load below
    # It would be nice if cms.Sequence would have .clear() or similar
    global earlyGeneralTracksSequence
    earlyGeneralTracksSequence.remove(earlyGeneralTracks)

    process.load("RecoTracker.FinalTrackSelectors.Phase1PU70_earlyGeneralTracks_cfi") # too different from Run2 that patching with era makes no sense

    earlyGeneralTracksSequence += process.earlyGeneralTracks

modifyRecoTrackerFinalTrackSelectorsEarlyGeneralTracksForPhase1Pixel_ = eras.phase1Pixel.makeProcessModifier(_modifyForPhase1)
