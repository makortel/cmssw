import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiPixelClusterizer.SiPixelClusterizer_cfi import SiPixelClusterizer as _SiPixelClusterizer
siPixelClusters = _SiPixelClusterizer.clone() # this file is a hack anyway and will get removed soon
siPixelClusters.src = 'simSiPixelDigis'
siPixelClusters.MissCalibrate = False

from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
siStripZeroSuppression.RawDigiProducersList = cms.VInputTag( cms.InputTag('simSiStripDigis','VirginRaw'),
                                                                     cms.InputTag('simSiStripDigis','ProcessedRaw'),
                                                                     cms.InputTag('simSiStripDigis','ScopeMode'))

from RecoLocalTracker.SiStripClusterizer.SiStripClusterizer_cfi import *
siStripClusters.DigiProducersList = cms.VInputTag(cms.InputTag('simSiStripDigis','ZeroSuppressed'),
                                                          cms.InputTag('siStripZeroSuppression','VirginRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ProcessedRaw'),
                                                          cms.InputTag('siStripZeroSuppression','ScopeMode'))
