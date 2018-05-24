import FWCore.ParameterSet.Config as cms
import EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi
import EventFilter.SiPixelRawToDigi.siPixelRawToDigiHeterogeneous_cfi

siPixelDigis = EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi.siPixelRawToDigi.clone()
siPixelDigis.Timing = cms.untracked.bool(False)
siPixelDigis.IncludeErrors = cms.bool(True)
siPixelDigis.InputLabel = cms.InputTag("siPixelRawData")
siPixelDigis.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
siPixelDigis.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
siPixelDigis.UserErrorList = cms.vint32(40)
##  Use pilot blades
siPixelDigis.UsePilotBlade = cms.bool(False)
##  Use phase1
siPixelDigis.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
siPixelDigis.Regions = cms.PSet( ) 
siPixelDigis.CablingMapLabel = cms.string("")

_siPixelDigisHeterogeneous = EventFilter.SiPixelRawToDigi.siPixelRawToDigiHeterogeneous_cfi.siPixelRawToDigiHeterogeneous.clone()
_siPixelDigisHeterogeneous.IncludeErrors = cms.bool(True)
_siPixelDigisHeterogeneous.InputLabel = cms.InputTag("rawDataCollector")
_siPixelDigisHeterogeneous.UseQualityInfo = cms.bool(False)
## ErrorList: list of error codes used by tracking to invalidate modules
_siPixelDigisHeterogeneous.ErrorList = cms.vint32(29)
## UserErrorList: list of error codes used by Pixel experts for investigation
_siPixelDigisHeterogeneous.UserErrorList = cms.vint32(40)
##  Use pilot blades
_siPixelDigisHeterogeneous.UsePilotBlade = cms.bool(False)
##  Use phase1
_siPixelDigisHeterogeneous.UsePhase1 = cms.bool(False)
## Empty Regions PSet means complete unpacking
_siPixelDigisHeterogeneous.Regions = cms.PSet( )
_siPixelDigisHeterogeneous.CablingMapLabel = cms.string("")

# The following is copied fro siPixelClusters_cfi, clearly not
# maintainable in the long run
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(_siPixelDigisHeterogeneous,
  VCaltoElectronGain      = cms.int32(47),   # L2-4: 47 +- 4.7
  VCaltoElectronGain_L1   = cms.int32(50),   # L1:   49.6 +- 2.6
  VCaltoElectronOffset    = cms.int32(-60),  # L2-4: -60 +- 130
  VCaltoElectronOffset_L1 = cms.int32(-670), # L1:   -670 +- 220
  ChannelThreshold        = cms.int32(10),
  SeedThreshold           = cms.int32(1000),
  ClusterThreshold        = cms.int32(4000),
  ClusterThreshold_L1     = cms.int32(2000)
)

from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toReplaceWith(siPixelDigis, _siPixelDigisHeterogeneous)

phase1Pixel.toModify(siPixelDigis, UsePhase1=True)
