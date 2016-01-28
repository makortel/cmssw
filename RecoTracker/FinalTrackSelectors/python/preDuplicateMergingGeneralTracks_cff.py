import FWCore.ParameterSet.Config as cms

from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *

preDuplicateMergingGeneralTracksSequence = cms.Sequence(
    preDuplicateMergingGeneralTracks
)

from Configuration.StandardSequences.Eras import eras
def _modifyForPhase1(process):
    # Need to clear because of the load below
    # It would be nice if cms.Sequence would have .clear() or similar
    global preDuplicateMergingGeneralTracksSequence
    preDuplicateMergingGeneralTracksSequence.remove(preDuplicateMergingGeneralTracks)

    process.load("RecoTracker.FinalTrackSelectors.Phase1PU70_preDuplicateMergingGeneralTracks_cfi") # too different from Run2 that patching with era makes no sense

    preDuplicateMergingGeneralTracksSequence += process.preDuplicateMergingGeneralTracks

modifyRecoTrackerFinalTrackSelectorsPreDuplicateMergingGeneralTracksForPhase1Pixel_ = eras.phase1Pixel.makeProcessModifier(_modifyForPhase1)
