import FWCore.ParameterSet.Config as cms


# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import *
pixelPairStepClusters = trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("lowPtTripletStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    oldClusterRemovalInfo                    = cms.InputTag("lowPtTripletStepClusters"),
    trackClassifier                          = cms.InputTag('lowPtTripletStep',"QualityMasks"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)

# SEEDING LAYERS
pixelPairStepSeedLayers = cms.EDProducer("SeedingLayersEDProducer",
    layerList = cms.vstring('BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3', 
        'BPix1+FPix1_pos', 'BPix1+FPix1_neg', 
        'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
        'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg'),
    BPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    ),
    FPix = cms.PSet(
        TTRHBuilder = cms.string('WithTrackAngle'),
        HitProducer = cms.string('siPixelRecHits'),
        skipClusters = cms.InputTag('pixelPairStepClusters')
    )
)

# SEEDS
import RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff
pixelPairStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromPairsWithVertices_cff.globalSeedsFromPairsWithVertices.clone()
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = cms.InputTag("firstStepPrimaryVertices")
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 0.6
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.originRadius = 0.015
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.fixedError = 0.03
pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.useMultipleScattering = True
pixelPairStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = cms.InputTag('pixelPairStepSeedLayers')

pixelPairStepSeeds.SeedComparitorPSet = cms.PSet(
        ComponentName = cms.string('PixelClusterShapeSeedComparitor'),
        FilterAtHelixStage = cms.bool(True),
        FilterPixelHits = cms.bool(True),
        FilterStripHits = cms.bool(False),
        ClusterShapeHitFilterName = cms.string('ClusterShapeHitFilter'),
        ClusterShapeCacheSrc = cms.InputTag('siPixelClusterShapeCache'),
    )

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff
pixelPairStepTrajectoryFilterBase = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    seedPairPenalty =0,
    minPt = 0.1,
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
pixelPairStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
pixelPairStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('pixelPairStepTrajectoryFilterShape'))
    ),
)



import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
pixelPairStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('pixelPairStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(9.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
    pTChargeCutThreshold = cms.double(15.)
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
pixelPairStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    MeasurementTrackerName = '',
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryFilter')),
    maxCand = 3,
    estimator = cms.string('pixelPairStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7) 
    )

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
pixelPairStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('pixelPairStepSeeds'),
    clustersToSkip = cms.InputTag('pixelPairStepClusters'),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('pixelPairStepTrajectoryBuilder')),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),

)


# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
pixelPairStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    AlgorithmName = cms.string('pixelPairStep'),
    src = 'pixelPairStepTrackCandidates',
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
pixelPairStep =  TrackMVAClassifierPrompt.clone()
pixelPairStep.src = 'pixelPairStepTracks'
pixelPairStep.GBRForestLabel = 'MVASelectorIter2_13TeV'
pixelPairStep.qualityCuts = [-0.2,0.0,0.3]


# Final sequence
PixelPairStep = cms.Sequence(pixelPairStepClusters*
                         pixelPairStepSeedLayers*
                         pixelPairStepSeeds*
                         pixelPairStepTrackCandidates*
                         pixelPairStepTracks*
                         pixelPairStep)

from Configuration.StandardSequences.Eras import eras
# Customization for phase1
def _modifyForPhase1(process):
    # Cluster mask
    pixelPairStepClusters.trajectories = "mixedTripletStepTracks"
    pixelPairStepClusters.oldClusterRemovalInfo ="mixedTripletStepClusters"
    del pixelPairStepClusters.trackClassifier
    pixelPairStepClusters.overrideTrkQuals = "mixedTripletStep"

    # Seeding layers
    pixelPairStepSeedLayers.layerList = ['BPix1+BPix2', 'BPix1+BPix3', 'BPix2+BPix3',
                                         'BPix2+BPix4', 'BPix3+BPix4',
                                         'BPix1+FPix1_pos', 'BPix1+FPix1_neg',
                                         'BPix2+FPix1_pos', 'BPix2+FPix1_neg', 
                                         'FPix1_pos+FPix2_pos', 'FPix1_neg+FPix2_neg',
                                         'FPix2_pos+FPix3_pos', 'FPix2_neg+FPix3_neg']

    # Seeding
    pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.ptMin = 1.2
    pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.useMultipleScattering = False
    pixelPairStepSeeds.SeedCreatorPSet.magneticField = ''
    pixelPairStepSeeds.SeedCreatorPSet.propagator = 'PropagatorWithMaterial'
    pixelPairStepSeeds.ClusterCheckPSet.doClusterCheck = False
    pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement = 0


    # Building quality cuts
    pixelPairStepTrajectoryFilterBase.maxLostHitsFraction = 1./10.
    pixelPairStepTrajectoryFilterBase.constantValueForLostHitsFractionFilter = 0.801
    pixelPairStepTrajectoryFilterBase.seedPairPenalty = TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.seedPairPenalty.value()
    import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
    pixelPairStepChi2Est.MaxChi2 = 16.0
    pixelPairStepChi2Est.clusterChargeCut = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clusterChargeCut.value()
    pixelPairStepChi2Est.pTChargeCutThreshold = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.pTChargeCutThreshold.value()

    # Track candidates
    from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits
    process.pixelPairStepTrajectoryCleanerBySharedHits = trajectoryCleanerBySharedHits.clone(
            ComponentName = cms.string('pixelPairStepTrajectoryCleanerBySharedHits'),
                fractionShared = cms.double(0.095),
                allowSharedFirstHit = cms.bool(True)
                )
    pixelPairStepTrackCandidates.TrajectoryCleaner = 'pixelPairStepTrajectoryCleanerBySharedHits'

    # Fitting
    pixelPairStepTracks.TTRHBuilder = 'WithTrackAngle'

    # Remove modules not used in Phase1PU70
    global PixelPairStep
    PixelPairStep.remove(pixelPairStep)

    # Then add the old-style cut-based track selector back
    import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
    process.pixelPairStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='pixelPairStepTracks',
        trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'pixelPairStepLoose',
                chi2n_par = 0.9,
                res_par = ( 0.003, 0.002 ),
                minNumberLayers = 3,
                maxNumberLostLayers = 3,
                minNumber3DLayers = 3,
                d0_par1 = ( 0.4, 4.0 ),
                dz_par1 = ( 0.4, 4.0 ),
                d0_par2 = ( 0.4, 4.0 ),
                dz_par2 = ( 0.4, 4.0 )
                ), #end of pset
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
                name = 'pixelPairStepTight',
                preFilterName = 'pixelPairStepLoose',
                chi2n_par = 0.6,
                res_par = ( 0.003, 0.002 ),
                minNumberLayers = 4,
                maxNumberLostLayers = 2,
                minNumber3DLayers = 3,
                d0_par1 = ( 0.3, 4.0 ),
                dz_par1 = ( 0.3, 4.0 ),
                d0_par2 = ( 0.3, 4.0 ),
                dz_par2 = ( 0.3, 4.0 )
                ),
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
                name = 'pixelPairStep',
                preFilterName = 'pixelPairStepTight',
                chi2n_par = 0.4,
                res_par = ( 0.003, 0.001 ),
                minNumberLayers = 4,
                maxNumberLostLayers = 1,
                minNumber3DLayers = 3,
                d0_par1 = ( 0.2, 4.0 ),
                dz_par1 = ( 0.25, 4.0 ),
                d0_par2 = ( 0.25, 4.0 ),
                dz_par2 = ( 0.25, 4.0 )
                ),
            ) #end of vpset
        ) #end of clone
    PixelPairStep += process.pixelPairStepSelector

modifyRecoTrackerIterativeTrackingPixelPairStepPhase1Pixel_ = eras.phase1Pixel.makeProcessModifier(_modifyForPhase1)
