import FWCore.ParameterSet.Config as cms

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
lowPtTripletStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("lowPtQuadStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("lowPtQuadStepClusters"),
    trackClassifier                          = cms.InputTag('lowPtQuadStep',"QualityMasks"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
    layerList = [
        'BPix1+BPix2+BPix3',
        'BPix2+BPix3+BPix4',
        'BPix1+BPix3+BPix4',
        'BPix1+BPix2+BPix4',
        'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
        'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
        'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
        'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
        'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
        'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
        'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
        'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
        'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
    ]
)
lowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
lowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.35, # FIXME: Phase1PU70 value, let's see if we can lower it to Run2 value (0.2)
            originRadius = 0.02,
            nSigmaZ = 4.0
        )
    )
)
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtTripletStepSeedLayers'
lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False) # FIXME: cluster check condition to be updated

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
lowPtTripletStepStandardTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
    maxCCCLostHits = 1,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtTripletStepTrajectoryFilter = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('lowPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
lowPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryFilter')),
    maxCand = 4,
    estimator = cms.string('lowPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
lowPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('lowPtTripletStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('lowPtTripletStepTrajectoryBuilder')),
    clustersToSkip = cms.InputTag('lowPtTripletStepClusters'),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
lowPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'lowPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('lowPtTripletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle') # FIXME: to be updated once we get the templates to GT
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.16),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtTripletStepTrackCandidates.TrajectoryCleaner = 'lowPtTripletStepTrajectoryCleanerBySharedHits'

# Final selection
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
#lowPtTripletStep =  TrackMVAClassifierPrompt.clone()
#lowPtTripletStep.src = 'lowPtTripletStepTracks'
#lowPtTripletStep.GBRForestLabel = 'MVASelectorIter1_13TeV'
#lowPtTripletStep.qualityCuts = [-0.6,-0.3,-0.1]

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier
lowPtTripletStep = TrackCutClassifier.clone(
    src = "lowPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
)
lowPtTripletStep.mva.minPixelHits = [1,1,1]
lowPtTripletStep.mva.maxChi2 = [9999.,9999.,9999.]
lowPtTripletStep.mva.maxChi2n = [2.0,0.9,0.5]
lowPtTripletStep.mva.minLayers = [3,3,3]
lowPtTripletStep.mva.min3DLayers = [3,3,3]
lowPtTripletStep.mva.maxLostLayers = [2,2,2]
lowPtTripletStep.mva.dr_par.dr_par1 = [0.8,0.7,0.6]
lowPtTripletStep.mva.dz_par.dz_par1 = [0.7,0.6,0.45]
lowPtTripletStep.mva.dr_par.dr_par2 = [0.5,0.4,0.3]
lowPtTripletStep.mva.dz_par.dz_par2 = [0.5,0.4,0.4]
lowPtTripletStep.mva.dr_par.d0err_par = [0.002,0.002,0.001]



# Final sequence
LowPtTripletStep = cms.Sequence(lowPtTripletStepClusters*
                                lowPtTripletStepSeedLayers*
                                lowPtTripletStepSeeds*
                                lowPtTripletStepTrackCandidates*
                                lowPtTripletStepTracks*
                                lowPtTripletStep)
