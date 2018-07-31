import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDACore.cudaDeviceChooserFilter_cfi import cudaDeviceChooserFilter as _cudaDeviceChooserFilter
prod5CUDADeviceFilter = _cudaDeviceChooserFilter.clone()
