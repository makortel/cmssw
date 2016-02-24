import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import SiPixelClusterizer as _SiPixelClusterizer
from RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *

# Minimal migration of label change in SiPixelClusterizer_cfi
# Never mix this file and RecoLocalTracker.Configuration.siPixelClusters_cfi
# in the same application!
siPixelClusters = _SiPixelClusterizer.clone()

pixeltrackerlocalreco = cms.Sequence(siPixelClusters*siPixelRecHits)
striptrackerlocalreco = cms.Sequence(siStripZeroSuppression*siStripClusters*siStripMatchedRecHits)
trackerlocalreco = cms.Sequence(pixeltrackerlocalreco*striptrackerlocalreco)

