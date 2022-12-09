import FWCore.ParameterSet.Config as cms

class AlpakaModuleTypeResolver:
    def __init__(self, accelerators):
        # first element is used as the default is nothing is set
        self._valid_backends = []
        if "gpu-nvidia" in accelerators:
            self._valid_backends.append("cuda_async")
        if "cpu" in accelerators:
            self._valid_backends.append("serial_sync")
        if len(self._valid_backends) == 0:
            raise Exception("The job was configured to use {} accelerators, but Alpaka does not support any of them.".format(", ".join(accelerators)))

    def pluginAndConfiguration(self):
        return ("AlpakaModuleTypeResolver", cms.untracked.PSet())

    def setModuleBackend(self, module):
        if module.type_().endswith("@alpaka"):
            defaultBackend = self._valid_backends[0]
            if hasattr(module, "alpaka"):
                if hasattr(module.alpaka, "backend"):
                    if module.alpaka.backend.value() not in self._valid_backends:
                        raise Exception("Module {} has the Alpaka backend set explicitly, but its accelerator is not available for the job. The following Alpaka backends are available for the job {}.".format(module.label_(), ", ".join(self._valid_backends)))
                else:
                    module.alpaka.backend = cms.untracked.string(defaultBackend)
            else:
                module.alpaka = cms.untracked.PSet(
                    backend = cms.untracked.string(defaultBackend)
                )

class ProcessAcceleratorAlpaka(cms.ProcessAccelerator):
    """ProcessAcceleratorAlpaka itself does not define or inspect
    availability of any accelerator devices. It merely sets up
    necessary Alpaka infrastructure based on the availability of
    accelerators that the concrete ProcessAccelerators (like
    ProcessAcceleratorCUDA) define.
    """
    def __init__(self):
        super(ProcessAcceleratorAlpaka,self).__init__()
    def moduleTypeResolver(self, accelerators):
        return AlpakaModuleTypeResolver(accelerators)
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
