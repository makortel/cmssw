import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

#
# Tracker Local Reco
# Initialize magnetic field
#
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.siPixelClusters_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizerPreSplitting_cfi import *
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SubCollectionProducers.clustersummaryproducer_cfi import *

pixeltrackerlocalreco = cms.Sequence(siPixelClustersPreSplitting*siPixelRecHitsPreSplitting)
eras.trackingPhase1.toReplaceWith(pixeltrackerlocalreco, cms.Sequence(
    siPixelClusters*siPixelRecHits
))

striptrackerlocalreco = cms.Sequence(siStripZeroSuppression*siStripClusters*siStripMatchedRecHits)
trackerlocalreco = cms.Sequence(pixeltrackerlocalreco*striptrackerlocalreco*clusterSummaryProducer)


