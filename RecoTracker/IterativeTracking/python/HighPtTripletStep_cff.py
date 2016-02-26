import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

### high-pT triplets ###

# NEW CLUSTERS (remove previously used clusters)
from RecoLocalTracker.SubCollectionProducers.trackClusterRemover_cfi import trackClusterRemover as _trackClusterRemover
_highPtTripletStepClustersBase = _trackClusterRemover.clone(
    maxChi2                                  = cms.double(9.0),
    trajectories                             = cms.InputTag("initialStepTracks"),
    pixelClusters                            = cms.InputTag("siPixelClusters"),
    stripClusters                            = cms.InputTag("siStripClusters"),
    TrackQuality                             = cms.string('highPurity'),
    minNumberOfLayersWithMeasBeforeFiltering = cms.int32(0),
)
highPtTripletStepClusters = _highPtTripletStepClustersBase.clone(
    trackClassifier                          = cms.InputTag('initialStep',"QualityMasks"),
)
eras.trackingPhase1PU70.toReplaceWith(highPtTripletStepClusters, _highPtTripletStepClustersBase.clone(
    overrideTrkQuals                         = "initialStepSelector:initialStep",
))


# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
highPtTripletStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone(
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
highPtTripletStepSeedLayers.BPix.skipClusters = cms.InputTag('highPtTripletStepClusters')
highPtTripletStepSeedLayers.FPix.skipClusters = cms.InputTag('highPtTripletStepClusters')

# SEEDS
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
highPtTripletStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
            ptMin = 0.6,
            originRadius = 0.02,
            nSigmaZ = 4.0
        )
    )
)
highPtTripletStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'highPtTripletStepSeedLayers'
eras.trackingPhase1PU70.toModify(highPtTripletStepSeeds, RegionFactoryPSet = dict(RegionPSet = dict(ptMin = 0.7)))

from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
highPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor
highPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False) # FIXME: cluster check condition to be updated

# QUALITY CUTS DURING TRACK BUILDING
import TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff as _TrajectoryFilter_cff
_highPtTripletStepTrajectoryFilterBase = _TrajectoryFilter_cff.CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
)
highPtTripletStepTrajectoryFilterBase = _highPtTripletStepTrajectoryFilterBase.clone(
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
)
eras.trackingPhase1PU70.toReplaceWith(highPtTripletStepTrajectoryFilterBase, _highPtTripletStepTrajectoryFilterBase)
highPtTripletStepTrajectoryFilter = _TrajectoryFilter_cff.CompositeTrajectoryFilter_block.clone(
    filters = [cms.PSet( refToPSet_ = cms.string('highPtTripletStepTrajectoryFilterBase'))]
)

import RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi
highPtTripletStepChi2Est = RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi.Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('highPtTripletStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
    pTChargeCutThreshold = cms.double(15.)
)
eras.trackingPhase1PU70.toModify(highPtTripletStepChi2Est,
    clusterChargeCut = dict(refToPSet_ = "SiStripClusterChargeCutNone")
)

# TRACK BUILDING
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
highPtTripletStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('highPtTripletStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    # 0.63 GeV is the maximum pT for a charged particle to loop within the 1.1m radius
    # of the outermost Tracker barrel layer (with B=3.8T)
    maxPtForLooperReconstruction = cms.double(0.7)
)
eras.trackingPhase1PU70.toModify(highPtTripletStepTrajectoryBuilder,
    MeasurementTrackerName = '',
    maxCand = 4,
)

# MAKING OF TRACK CANDIDATES
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
highPtTripletStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('highPtTripletStepSeeds'),
    clustersToSkip = cms.InputTag('highPtTripletStepClusters'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('highPtTripletStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
)

# For Phase1PU70
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import trajectoryCleanerBySharedHits as _trajectoryCleanerBySharedHits
highPtTripletStepTrajectoryCleanerBySharedHits = _trajectoryCleanerBySharedHits.clone(
    ComponentName = 'highPtTripletStepTrajectoryCleanerBySharedHits',
    fractionShared = 0.16,
    allowSharedFirstHit = True
)
eras.trackingPhase1PU70.toModify(highPtTripletStepTrackCandidates, TrajectoryCleaner = 'highPtTripletStepTrajectoryCleanerBySharedHits')

# TRACK FITTING
import RecoTracker.TrackProducer.TrackProducer_cfi
highPtTripletStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'highPtTripletStepTrackCandidates',
    AlgorithmName = cms.string('highPtTripletStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother'),
    TTRHBuilder=cms.string('WithTrackAngle') # FIXME: to be updated once we get the templates to GT
)

# Final selection
# MVA selection to be enabled after re-training, for time being we go with cut-based selector
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
#from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *
#
#highPtTripletStepClassifier1 = TrackMVAClassifierPrompt.clone()
#highPtTripletStepClassifier1.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
#highPtTripletStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]
#
#from RecoTracker.IterativeTracking.Phase1_DetachedTripletStep_cff import detachedTripletStepClassifier1
#from RecoTracker.IterativeTracking.Phase1_LowPtTripletStep_cff import lowPtTripletStep
#highPtTripletStepClassifier2 = detachedTripletStepClassifier1.clone()
#highPtTripletStepClassifier2.src = 'highPtTripletStepTracks'
#highPtTripletStepClassifier3 = lowPtTripletStep.clone()
#highPtTripletStepClassifier3.src = 'highPtTripletStepTracks'
#
#
#from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
#highPtTripletStep = ClassifierMerger.clone()
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']
#highPtTripletStep.inputClassifiers=['highPtTripletStepClassifier1','highPtTripletStepClassifier2','highPtTripletStepClassifier3']

from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cfi import TrackCutClassifier
highPtTripletStep = TrackCutClassifier.clone(
    src = "highPtTripletStepTracks",
    vertices = "firstStepPrimaryVertices",
)
highPtTripletStep.mva.minPixelHits = [1,1,1]
highPtTripletStep.mva.maxChi2 = [9999.,9999.,9999.]
highPtTripletStep.mva.maxChi2n = [2.0,1.0,0.7]
highPtTripletStep.mva.minLayers = [3,3,3]
highPtTripletStep.mva.min3DLayers = [3,3,3]
highPtTripletStep.mva.maxLostLayers = [3,2,2]
highPtTripletStep.mva.dz_par.dz_par1 = [0.8,0.7,0.7]
highPtTripletStep.mva.dz_par.dz_par2 = [0.6,0.5,0.4]
highPtTripletStep.mva.dz_par.dz_exp = [4,4,4]
highPtTripletStep.mva.dr_par.dr_par1 = [0.7,0.6,0.5]
highPtTripletStep.mva.dr_par.dr_par2 = [0.4,0.35,0.25]
highPtTripletStep.mva.dr_par.dr_exp = [4,4,4]
highPtTripletStep.mva.dr_par.d0err_par = [0.002,0.002,0.001]

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
highPtTripletStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
    src = 'highPtTripletStepTracks',
    trackSelectors = cms.VPSet(
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
            name = 'highPtTripletStepLoose',
            chi2n_par = 2.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 3,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.7, 4.0 ),
            dz_par1 = ( 0.8, 4.0 ),
            d0_par2 = ( 0.4, 4.0 ),
            dz_par2 = ( 0.6, 4.0 )
        ), #end of pset
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.tightMTS.clone(
            name = 'highPtTripletStepTight',
            preFilterName = 'highPtTripletStepLoose',
            chi2n_par = 1.0,
            res_par = ( 0.003, 0.002 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.6, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.35, 4.0 ),
            dz_par2 = ( 0.5, 4.0 )
        ),
        RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.highpurityMTS.clone(
            name = 'highPtTripletStep',
            preFilterName = 'highPtTripletStepTight',
            chi2n_par = 0.7,
            res_par = ( 0.003, 0.001 ),
            minNumberLayers = 3,
            maxNumberLostLayers = 2,
            minNumber3DLayers = 3,
            d0_par1 = ( 0.5, 4.0 ),
            dz_par1 = ( 0.7, 4.0 ),
            d0_par2 = ( 0.25, 4.0 ),
            dz_par2 = ( 0.4, 4.0 )
        ),
    ) #end of vpset
) #end of clone

# Final sequence
HighPtTripletStep = cms.Sequence(highPtTripletStepClusters*
                                 highPtTripletStepSeedLayers*
                                 highPtTripletStepSeeds*
                                 highPtTripletStepTrackCandidates*
                                 highPtTripletStepTracks*
#                                 highPtTripletStepClassifier1*highPtTripletStepClassifier2*highPtTripletStepClassifier3*
                                 highPtTripletStep)
_HighPtTripletStep_Phase1PU70 = HighPtTripletStep.copy()
_HighPtTripletStep_Phase1PU70.replace(highPtTripletStep, highPtTripletStepSelector)
eras.trackingPhase1PU70.toReplaceWith(HighPtTripletStep, _HighPtTripletStep_Phase1PU70)
