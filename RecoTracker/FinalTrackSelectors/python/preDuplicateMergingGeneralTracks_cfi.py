import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

preDuplicateMergingGeneralTracks = TrackCollectionMerger.clone()
preDuplicateMergingGeneralTracks.trackProducers = [
    "earlyGeneralTracks", 
    "muonSeededTracksInOut",
    "muonSeededTracksOutIn",
    ]
preDuplicateMergingGeneralTracks.inputClassifiers =[
   "earlyGeneralTracks", 
   "muonSeededTracksInOutClassifier",
   "muonSeededTracksOutInClassifier"
   ]

preDuplicateMergingGeneralTracks.foundHitBonus  = 100.0
preDuplicateMergingGeneralTracks.lostHitPenalty =   1.0


# For Phase1PU70
import RecoTracker.FinalTrackSelectors.Phase1PU70_earlyGeneralTracks_cfi
_preDuplicateMergingGeneralTracks_Phase1PU70 =  RecoTracker.FinalTrackSelectors.Phase1PU70_earlyGeneralTracks_cfi.earlyGeneralTracks.clone(
    TrackProducers = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks"),
        cms.InputTag("muonSeededTracksInOut"),
        cms.InputTag("muonSeededTracksOutIn"),
    ),
    hasSelector = cms.vint32(0,1,1),
    selectedTrackQuals = cms.VInputTag(
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"), # not used but needed
        cms.InputTag("muonSeededTracksInOutSelector","muonSeededTracksInOutHighPurity"),
        cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity"),
    ),
    mvaValueTags = cms.VInputTag(
        cms.InputTag("earlyGeneralTracks","MVAVals"),
        cms.InputTag("muonSeededTracksInOutSelector","MVAVals"),
        cms.InputTag("muonSeededTracksOutInSelector","MVAVals"),
    ),
    setsToMerge = cms.VPSet(cms.PSet(pQual = cms.bool(True), tLists = cms.vint32(0, 1,2))),
    FoundHitBonus  = 100.0,
    LostHitPenalty =   1.0,
)
eras.phase1Pixel.toReplaceWith(preDuplicateMergingGeneralTracks, _preDuplicateMergingGeneralTracks_Phase1PU70)
