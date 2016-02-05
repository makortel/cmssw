import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

### STEP 0 ###

# hit building
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *

# SEEDING LAYERS
import RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi
initialStepSeedLayers = RecoTracker.TkSeedingLayers.PixelLayerTriplets_cfi.PixelLayerTriplets.clone()


# seeding
from RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff import *
from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
initialStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
    RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
    ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
    RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
    ptMin = 0.6,
    originRadius = 0.02,
    nSigmaZ = 4.0
    )
    )
    )
initialStepSeeds.OrderedHitsFactoryPSet.SeedingLayers = 'initialStepSeedLayers'
eras.phase1Pixel.toModify(initialStepSeeds.RegionFactoryPSet.RegionPSet, ptMin = 0.7)
eras.phase1Pixel.toModify(initialStepSeeds.SeedCreatorPSet, magneticField = '', propagator = 'PropagatorWithMaterial')
eras.phase1Pixel.toModify(initialStepSeeds.ClusterCheckPSet, doClusterCheck = False)
eras.phase1Pixel.toModify(initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet, maxElement = 0)
eras.phase1Pixel.toModify(initialStepSeeds, SeedMergerPSet = cms.PSet(
    layerList = cms.PSet(refToPSet_ = cms.string("PixelSeedMergerQuadruplets")),
    addRemainingTriplets = cms.bool(False),
    mergeTriplets = cms.bool(True),
    ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
))


from RecoPixelVertexing.PixelLowPtUtilities.ClusterShapeHitFilterESProducer_cfi import *
import RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi
initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet = RecoPixelVertexing.PixelLowPtUtilities.LowPtClusterShapeSeedComparitor_cfi.LowPtClusterShapeSeedComparitor

# building
from TrackingTools.TrajectoryFiltering.TrajectoryFilter_cff import CkfBaseTrajectoryFilter_block as _CkfBaseTrajectoryFilter_block
initialStepTrajectoryFilterBase = _CkfBaseTrajectoryFilter_block.clone(
    minimumNumberOfHits = 3,
    minPt = 0.2,
    maxCCCLostHits = 2,
    minGoodStripCharge = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutLoose'))
    )
eras.phase1Pixel.toModify(initialStepTrajectoryFilterBase,
                          maxCCCLostHits = _CkfBaseTrajectoryFilter_block.maxCCCLostHits.value(),
                          minGoodStripCharge = _CkfBaseTrajectoryFilter_block.minGoodStripCharge.clone())


import RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi
initialStepTrajectoryFilterShape = RecoPixelVertexing.PixelLowPtUtilities.StripSubClusterShapeTrajectoryFilter_cfi.StripSubClusterShapeTrajectoryFilterTIX12.clone()
initialStepTrajectoryFilter = cms.PSet(
    ComponentType = cms.string('CompositeTrajectoryFilter'),
    filters = cms.VPSet(
        cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterBase')),
    #    cms.PSet( refToPSet_ = cms.string('initialStepTrajectoryFilterShape'))
    ),
)

from RecoTracker.MeasurementDet.Chi2ChargeMeasurementEstimator_cfi import Chi2ChargeMeasurementEstimator as _Chi2ChargeMeasurementEstimator
initialStepChi2Est = _Chi2ChargeMeasurementEstimator.clone(
    ComponentName = cms.string('initialStepChi2Est'),
    nSigma = cms.double(3.0),
    MaxChi2 = cms.double(30.0),
    clusterChargeCut = cms.PSet(refToPSet_ = cms.string('SiStripClusterChargeCutTiny')),
    pTChargeCutThreshold = cms.double(15.)
)
eras.phase1Pixel.toModify(initialStepChi2Est,
                          clusterChargeCut = _Chi2ChargeMeasurementEstimator.clusterChargeCut.value(),
                          pTChargeCutThreshold = _Chi2ChargeMeasurementEstimator.pTChargeCutThreshold.value())

import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
initialStepTrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder.clone(
    trajectoryFilter = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryFilter')),
    alwaysUseInvalidHits = True,
    maxCand = 3,
    estimator = cms.string('initialStepChi2Est'),
    maxDPhiForLooperReconstruction = cms.double(2.0),
    maxPtForLooperReconstruction = cms.double(0.7)
    )
eras.phase1Pixel.toModify(initialStepTrajectoryBuilder, maxCand = 6)

import RecoTracker.CkfPattern.CkfTrackCandidates_cfi
initialStepTrackCandidates = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    src = cms.InputTag('initialStepSeeds'),
    ### these two parameters are relevant only for the CachingSeedCleanerBySharedInput
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    TrajectoryBuilderPSet = cms.PSet(refToPSet_ = cms.string('initialStepTrajectoryBuilder')),
    doSeedingRegionRebuilding = True,
    useHitsSplitting = True
    )

# fitting
import RecoTracker.TrackProducer.TrackProducer_cfi
initialStepTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'initialStepTrackCandidates',
    AlgorithmName = cms.string('initialStep'),
    Fitter = cms.string('FlexibleKFFittingSmoother')
    )
eras.phase1Pixel.toModify(initialStepTracks, TTRHBuilder = 'WithTrackAngle')


#vertices
import RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi
firstStepPrimaryVertices=RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi.offlinePrimaryVertices.clone()
firstStepPrimaryVertices.TrackLabel = cms.InputTag("initialStepTracks")
firstStepPrimaryVertices.vertexCollections = cms.VPSet(
     [cms.PSet(label=cms.string(""),
               algorithm=cms.string("AdaptiveVertexFitter"),
               minNdof=cms.double(0.0),
               useBeamConstraint = cms.bool(False),
               maxDistanceToBeam = cms.double(1.0)
               )
      ]
    )
 

# Final selection
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierPrompt_cfi import *
from RecoTracker.FinalTrackSelectors.TrackMVAClassifierDetached_cfi import *

initialStepClassifier1 = TrackMVAClassifierPrompt.clone()
initialStepClassifier1.src = 'initialStepTracks'
initialStepClassifier1.GBRForestLabel = 'MVASelectorIter0_13TeV'
initialStepClassifier1.qualityCuts = [-0.9,-0.8,-0.7]

from RecoTracker.IterativeTracking.DetachedTripletStep_cff import detachedTripletStepClassifier1
from RecoTracker.IterativeTracking.LowPtTripletStep_cff import lowPtTripletStep
initialStepClassifier2 = detachedTripletStepClassifier1.clone()
initialStepClassifier2.src = 'initialStepTracks'
initialStepClassifier3 = lowPtTripletStep.clone()
initialStepClassifier3.src = 'initialStepTracks'



from RecoTracker.FinalTrackSelectors.ClassifierMerger_cfi import *
initialStep = ClassifierMerger.clone()
initialStep.inputClassifiers=['initialStepClassifier1','initialStepClassifier2','initialStepClassifier3']

# For Phase1PU70
import RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi
initialStepSelector = RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.multiTrackSelector.clone(
        src='initialStepTracks',
        trackSelectors= cms.VPSet(
            RecoTracker.FinalTrackSelectors.multiTrackSelector_cfi.looseMTS.clone(
                name = 'initialStepLoose',
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
                name = 'initialStepTight',
                preFilterName = 'initialStepLoose',
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
                name = 'initialStep',
                preFilterName = 'initialStepTight',
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
InitialStep = cms.Sequence(initialStepSeedLayers*
                           initialStepSeeds*
                           initialStepTrackCandidates*
                           initialStepTracks*
                           firstStepPrimaryVertices*
                           initialStepClassifier1*initialStepClassifier2*initialStepClassifier3*
                           initialStep)
eras.phase1Pixel.toReplaceWith(InitialStep, cms.Sequence(
    initialStepSeedLayers*
    initialStepSeeds*
    initialStepTrackCandidates*
    initialStepTracks*
    initialStepSelector
))
