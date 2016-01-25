import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.InitialStepPreSplitting_cff import *
from RecoTracker.IterativeTracking.InitialStep_cff import *
from RecoTracker.IterativeTracking.DetachedTripletStep_cff import *
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelPairStep_cff import *
from RecoTracker.IterativeTracking.MixedTripletStep_cff import *
from RecoTracker.IterativeTracking.PixelLessStep_cff import *
from RecoTracker.IterativeTracking.TobTecStep_cff import *
from RecoTracker.IterativeTracking.JetCoreRegionalStep_cff import *

from RecoTracker.FinalTrackSelectors.earlyGeneralTracks_cfi import *
from RecoTracker.IterativeTracking.MuonSeededStep_cff import *
from RecoTracker.FinalTrackSelectors.preDuplicateMergingGeneralTracks_cfi import *
from RecoTracker.FinalTrackSelectors.MergeTrackCollections_cff import *
from RecoTracker.ConversionSeedGenerators.ConversionStep_cff import *

iterTracking = cms.Sequence(InitialStepPreSplitting*
                            InitialStep*
                            DetachedTripletStep*
                            LowPtTripletStep*
                            PixelPairStep*
                            MixedTripletStep*
                            PixelLessStep*
                            TobTecStep*
			    JetCoreRegionalStep *	
                            earlyGeneralTracks*
                            muonSeededStep*
                            preDuplicateMergingGeneralTracks*
                            generalTracksSequence*
                            ConvStep*
                            conversionStepTracks
                            )

from Configuration.StandardSequences.Eras import eras
def _modifyForPhase1(process):
    # It is more clear to fully clear and build iterTracking sequence
    # than to remove+add invidiual parts of it
    # It would be nice if cms.Sequence would have .clear() or similar
    global iterTracking
    iterTracking.remove(InitialStepPreSplitting)
    iterTracking.remove(InitialStep)
    iterTracking.remove(DetachedTripletStep)
    iterTracking.remove(LowPtTripletStep)
    iterTracking.remove(PixelPairStep)
    iterTracking.remove(MixedTripletStep)
    iterTracking.remove(PixelLessStep)
    iterTracking.remove(TobTecStep) 
    iterTracking.remove(JetCoreRegionalStep)
    iterTracking.remove(earlyGeneralTracks) 
    iterTracking.remove(muonSeededStep) 
    iterTracking.remove(preDuplicateMergingGeneralTracks)
    iterTracking.remove(generalTracksSequence) 
    iterTracking.remove(ConvStep) 
    iterTracking.remove(conversionStepTracks) 

    # Need to clear this sequence too because of the loads below
    # It would be nice if cms.Sequence would have .clear() or similar
    TobTecStep.remove(tobTecStepClusters)
    TobTecStep.remove(tobTecStepSeedLayersTripl)
    TobTecStep.remove(tobTecStepSeedsTripl)
    TobTecStep.remove(tobTecStepSeedLayersPair)
    TobTecStep.remove(tobTecStepSeedsPair)
    TobTecStep.remove(tobTecStepSeeds)
    TobTecStep.remove(tobTecStepTrackCandidates)
    TobTecStep.remove(tobTecStepTracks)
    TobTecStep.remove(tobTecStepClassifier1)
    TobTecStep.remove(tobTecStepClassifier2)
    TobTecStep.remove(tobTecStep)

    process.load("RecoTracker.IterativeTracking.Phase1PU70_HighPtTripletStep_cff")
    process.load("RecoTracker.IterativeTracking.Phase1PU70_LowPtQuadStep_cff")
    process.load("RecoTracker.IterativeTracking.Phase1PU70_DetachedQuadStep_cff")
    process.load("RecoTracker.IterativeTracking.Phase1PU70_TobTecStep_cff") # too different from Run2 that patching with era makes no sense
    process.load("RecoTracker.FinalTrackSelectors.Phase1PU70_earlyGeneralTracks_cfi") # too different from Run2 that patching with era makes no sense
    process.load("RecoTracker.FinalTrackSelectors.Phase1PU70_preDuplicateMergingGeneralTracks_cfi") # too different from Run2 that patching with era makes no sense

    # Build new sequence (need to use the existing Sequence object)
    iterTracking += (InitialStep +
                     process.HighPtTripletStep +
                     process.LowPtQuadStep +
                     LowPtTripletStep +
                     process.DetachedQuadStep +
                     MixedTripletStep +
                     PixelPairStep +
                     process.TobTecStep +
                     process.earlyGeneralTracks +
                     muonSeededStep +
                     process.preDuplicateMergingGeneralTracks +
                     generalTracksSequence +
                     ConvStep +
                     conversionStepTracks)

modifyRecoTrackerIterativeTrackingIterativeTkForPhase1Pixel_ = eras.phase1Pixel.makeProcessModifier(_modifyForPhase1)
