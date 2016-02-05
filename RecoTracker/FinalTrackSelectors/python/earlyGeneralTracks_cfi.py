import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks =  TrackCollectionMerger.clone()
earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                     'jetCoreRegionalStepTracks',
                                     'lowPtTripletStepTracks',
                                     'pixelPairStepTracks',
                                     'detachedTripletStepTracks',
                                     'mixedTripletStepTracks',
                                     'pixelLessStepTracks',
                                     'tobTecStepTracks'
                                     ]
earlyGeneralTracks.inputClassifiers =["initialStep",
                                      "jetCoreRegionalStep",
                                      "lowPtTripletStep",
                                      "pixelPairStep",
                                      "detachedTripletStep",
                                      "mixedTripletStep",
                                      "pixelLessStep",
                                      "tobTecStep"
                                      ]

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
_earlyGeneralTracks_Phase1PU70 = RecoTracker.FinalTrackSelectors.trackListMerger_cfi.trackListMerger.clone(
    TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('highPtTripletStepTracks'),
                      cms.InputTag('lowPtQuadStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('detachedQuadStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'),
                      cms.InputTag('pixelPairStepTracks'),
                      cms.InputTag('tobTecStepTracks')),
    hasSelector=cms.vint32(1,1,1,1,1,1,1,1),
    indivShareFrac=cms.vdouble(1.0,0.16,0.095,0.09,0.095,0.095,0.095,0.08),
    selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
                                       cms.InputTag("lowPtQuadStepSelector","lowPtQuadStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("detachedQuadStep"),
                                       cms.InputTag("mixedTripletStep"),
                                       cms.InputTag("pixelPairStepSelector","pixelPairStep"),
                                       cms.InputTag("tobTecStepSelector","tobTecStep")
                                       ),
    setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4,5,6,7), pQual=cms.bool(True) )
                             ),
    copyExtras = True,
    makeReKeyedSeeds = cms.untracked.bool(False)
)
eras.phase1Pixel.toReplaceWith(earlyGeneralTracks, _earlyGeneralTracks_Phase1PU70)
