import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_lowPtTripletStepClustersBase = _trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("detachedTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("detachedTripletStepClusters"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)
lowPtTripletStepClusters = _lowPtTripletStepClustersBase.clone(
    trackClassifier                          = cms.InputTag('detachedTripletStep',"QualityMasks"),
)
eras.trackingPhase1PU70.toReplaceWith(lowPtTripletStepClusters, _lowPtTripletStepClustersBase.clone(
    trajectories                             = "lowPtQuadStepTracks",
    oldClusterRemovalInfo                    = "lowPtQuadStepClusters",
    overrideTrkQuals                         = "lowPtQuadStepSelector:lowPtQuadStep",
))

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
lowPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()
lowPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
lowPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('lowPtTripletStepClusters')
eras.trackingPhase1PU70.toModify(lowPtTripletStepSeedLayers, layerList = [
    'BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
    'BPix1+BPix3+BPix4', 'BPix1+BPix2+BPix4',
    'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
    'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
    'BPix1+BPix3+FPix1_pos', 'BPix1+BPix3+FPix1_neg',
    'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
    'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
    'BPix1+BPix2+FPix2_pos', 'BPix1+BPix2+FPix2_neg',
    'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
    'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
    'BPix1+FPix1_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix3_neg'
])

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
lowPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.2,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'lowPtTripletStepSeedLayers'
eras.trackingPhase1PU70.toModify(lowPtTripletStepSeeds,
    RegionFactoryPSet = dict(
        RegionPSet = dict(
            ptMin = 0.35,
            originRadius = 0.015
        )
    ),
)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor


# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_lowPtTripletStepStandardTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.075,
)
lowPtTripletStepStandardTrajectoryFilter = _lowPtTripletStepStandardTrajectoryFilterBase.clone(
    maxCCCLostHits = 1,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
eras.trackingPhase1PU70.toReplaceWith(lowPtTripletStepStandardTrajectoryFilter, _lowPtTripletStepStandardTrajectoryFilterBase)

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeTrajectoryFilter_cfi import *
# Composite filter
lowPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters   = [cms.PSet(refToPSet_ = cms.string('lowPtTripletStepStandardTrajectoryFilter')),
                 # cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))
                ]
    )
eras.trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryFilter,
    filters = lowPtTripletStepTrajectoryFilter.filters + [cms.PSet(refToPSet_ = cms.string('ClusterShapeTrajectoryFilter'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
lowPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('lowPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
)
eras.trackingPhase1PU70.toModify(lowPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = 'SiStripClusterChargeCutNone'),
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
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
lowPtTripletStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
        ComponentName = cms.string('lowPtTripletStepTrajectoryCleanerBySharedHits'),
            fractionShared = cms.double(0.16),
            allowSharedFirstHit = cms.bool(True)
            )
lowPtTripletStepTrackCandidates.TrajectoryCleaner = 'lowPtTripletStepTrajectoryCleanerBySharedHits'
eras.trackingPhase1PU70.toModify(lowPtTripletStepTrajectoryCleanerBySharedHits, fractionShared = 0.09)

# Final selection



from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
lowPtTripletStep =  TrackMVAClassifierPrompt.clone()
lowPtTripletStep.src = 'lowPtTripletStepTracks'
lowPtTripletStep.GBRForestLabel = 'MVASelectorIter1_13TeV'
lowPtTripletStep.qualityCuts = [-0.6,-0.3,-0.1]

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
lowPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'lowPtTripletStepTracks',
    trackSelectors = [
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'lowPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.8, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.5, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'lowPtTripletStepTight',
            preFilterName = 'lowPtTripletStepLoose',
            chi2n_par = 0.9,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.6, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'lowPtTripletStep',
            preFilterName = 'lowPtTripletStepTight',
            chi2n_par = 0.5,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.45, 4.0 ),
            d0_par2 = ( 0.3, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ] #end of vpset
) #end of clone


# Final sequence
LowPtTripletStep = cms.Sequence(lowPtTripletStepClusters*
                                lowPtTripletStepSeedLayers*
                                lowPtTripletStepSeeds*
                                lowPtTripletStepTrackCandidates*
                                lowPtTripletStepTracks*
                                lowPtTripletStep)
_LowPtTripletStep_Phase1PU70 = LowPtTripletStep.copy()
_LowPtTripletStep_Phase1PU70.replace(lowPtTripletStep, lowPtTripletStepSelector)
eras.trackingPhase1PU70.toReplaceWith(LowPtTripletStep, _LowPtTripletStep_Phase1PU70)
