import FWCore.ParameterSet.Config as cms
from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import *

import RecoTracker.FinalTrackSelectors.trackListMerger_cfi
earlyGeneralTracks =  TrackCollectionMerger.clone()
earlyGeneralTracks.trackProducers = ['initialStepTracks',
                                     'highPtTripletStepTracks',
                                     'jetCoreRegionalStepTracks',
                                     'lowPtQuadStepTracks',
                                     'lowPtTripletStepTracks',
                                     'detachedQuadStepTracks',
                                     'detachedTripletStepTracks',
                                     'mixedTripletStepTracks',
                                     'pixelLessStepTracks',
                                     'tobTecStepTracks'
                                     ]
earlyGeneralTracks.inputClassifiers =["initialStep",
                                      "highPtTripletStep",
                                      "jetCoreRegionalStep",
                                      "lowPtQuadStep",
                                      "lowPtTripletStep",
                                      "detachedQuadStep",
                                      "detachedTripletStep",
                                      "mixedTripletStep",
                                      "pixelLessStep",
                                      "tobTecStep"
                                      ]
