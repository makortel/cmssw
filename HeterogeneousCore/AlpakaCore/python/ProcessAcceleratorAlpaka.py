import FWCore.ParameterSet.Config as cms

class ProcessAcceleratorAlpaka(cms.ProcessAccelerator):
    """ProcessAcceleratorAlpaka itself does not define or inspect
    availability of any accelerator devices. It merely sets up
    necessary Alpaka infrastructure based on the availability of
    accelerators that the concrete ProcessAccelerators (like
    ProcessAcceleratorCUDA) define.
    """
    def __init__(self):
        super(ProcessAcceleratorAlpaka,self).__init__()
    def apply(self, process, accelerators):
        if not hasattr(process, "AlpakaServiceSerialSync"):
            from HeterogeneousCore.AlpakaServices.AlpakaServiceSerialSync_cfi import AlpakaServiceSerialSync
            process.add_(AlpakaServiceSerialSync)
        if not hasattr(process, "AlpakaServiceCudaAsync"):
            from HeterogeneousCore.AlpakaServices.AlpakaServiceCudaAsync_cfi import AlpakaServiceCudaAsync
            process.add_(AlpakaServiceCudaAsync)

        if not hasattr(process.MessageLogger, "AlpakaService"):
            process.MessageLogger.AlpakaService = cms.untracked.PSet()

        process.AlpakaServiceSerialSync.enabled = "cpu" in accelerators
        process.AlpakaServiceCudaAsync.enabled = "gpu-nvidia" in accelerators
            
cms.specialImportRegistry.registerSpecialImportForType(ProcessAcceleratorAlpaka, "from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka import ProcessAcceleratorAlpaka")
