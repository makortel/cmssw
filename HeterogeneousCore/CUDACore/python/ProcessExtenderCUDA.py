import FWCore.ParameterSet.Config as cms

import os

class ProcessExtenderCUDA(cms.ProcessExtender):
    def __init__(self):
        super(ProcessExtenderCUDA,self).__init__()
    def isEnabled(self):
        return (os.system("cudaIsEnabled") == 0)
    def label(self):
        return "gpu-nvidia"
    def apply(self, process):
        if not hasattr(process, "CUDAService"):
            process.load("HeterogeneousCore.CUDAServices.CUDAService_cfi")

        if self.label() in process.options.accelerators:
            process.CUDAService.enabled = True
            process.MessageLogger.CUDAService = cms.untracked.PSet()
        else:
            process.CUDAService.enabled = False
            
cms.specialImportRegistry.registerSpecialImportForType(ProcessExtenderCUDA, "from HeterogeneousCore.CUDACore.ProcessExtenderCUDA import ProcessExtenderCUDA")
