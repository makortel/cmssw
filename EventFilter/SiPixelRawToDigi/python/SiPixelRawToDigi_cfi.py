import FWCore.ParameterSet.Config as cms
from EventFilter.SiPixelRawToDigi.siPixelRawToDigi_cfi import siPixelRawToDigi as _siPixelRawToDigi

from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA
siPixelDigis = SwitchProducerCUDA(
    cpu = _siPixelRawToDigi.clone()
)

from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(siPixelDigis.cpu, UsePhase1=True)

from EventFilter.SiPixelRawToDigi.siPixelDigisFromSoA_cfi import siPixelDigisFromSoA as _siPixelDigisFromSoA
from Configuration.ProcessModifiers.gpu_cff import gpu
gpu.toModify(siPixelDigis,
    cuda = _siPixelDigisFromSoA.clone(digiSrc = "siPixelClustersPreSplitting")
)
