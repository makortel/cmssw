import FWCore.ParameterSet.Config as cms

from HeterogeneousCore.CUDATest.prod5CUDADeviceFilter_cfi import prod5CUDADeviceFilter

# The prod6 is the final, (legacy) CPU-only product name, and the
# prod6Task is the Task containing all modules. The function itself
# sets up everything else.
from HeterogeneousCore.CUDATest.setupHeterogeneous import setupCUDA
(prod5, prod5Task) = setupCUDA("prod5", prod5CUDADeviceFilter, globals())
