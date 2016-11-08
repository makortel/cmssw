import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import *
from RecoPixelVertexing.PixelTriplets.PixelTripletHLTGenerator_cfi import *


PixelTrackReconstructionBlock = cms.PSet (
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
    Filter = cms.InputTag("pixelTrackFilterByKinematics"),
    RegionFactoryPSet = cms.PSet(
        RegionPsetFomBeamSpotBlock,
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot')
    ),
    OrderedHitsFactoryPSet = cms.PSet(
        ComponentName = cms.string('StandardHitTripletGenerator'),
        SeedingLayers = cms.InputTag('PixelLayerTriplets'),
        GeneratorPSet = cms.PSet(
            PixelTripletHLTGeneratorWithFilter
        )
    ),
    CleanerPSet = cms.PSet(
        ComponentName = cms.string('PixelTrackCleanerBySharedHits'),
        useQuadrupletAlgo = cms.bool(False),
    )
)

_OrderedHitsFactoryPSet_LowPU_Phase1PU70 = dict(
    SeedingLayers = "PixelLayerTripletsPreSplitting",
    GeneratorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"))
)
eras.trackingLowPU.toModify(PixelTrackReconstructionBlock, OrderedHitsFactoryPSet = _OrderedHitsFactoryPSet_LowPU_Phase1PU70)
eras.trackingPhase1PU70.toModify(PixelTrackReconstructionBlock,
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string('PixelSeedMergerQuadruplets')),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    RegionFactoryPSet = dict(RegionPSet = dict(originRadius =  0.02)),
    OrderedHitsFactoryPSet = _OrderedHitsFactoryPSet_LowPU_Phase1PU70,
)
eras.trackingPhase2PU140.toModify(PixelTrackReconstructionBlock,
    SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string('PixelSeedMergerQuadruplets')),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    ),
    RegionFactoryPSet = dict(RegionPSet = dict(originRadius =  0.02)),
    OrderedHitsFactoryPSet = dict(
      SeedingLayers = "PixelLayerTripletsPreSplitting",
      GeneratorPSet = dict(SeedComparitorPSet = dict(clusterShapeCacheSrc = "siPixelClusterShapeCachePreSplitting"),
                           maxElement = 0)
    )
)

