
import FWCore.ParameterSet.Config as cms

#
from CondTools.SiPixel.SiPixelGainCalibrationService_cfi import *
from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import SiPixelClusterizer as _SiPixelClusterizer
siPixelClustersPreSplitting = _SiPixelClusterizer.clone()
