import FWCore.ParameterSet.Config as cms

# Prototype of the function
import importlib
def setupHeterogeneous(prefix, deviceTypes, deviceFilters, modDict,
                       package=None, transferModuleNames={}, fallbackModuleName=None):
    """
    Mandatory parameters:
    prefix   -- common prefix of the CPU, CUDA, etc producers
    deviceTypes -- list of strings for the device types
    deviceFilters -- dict of non-CPU device types to device filter modules
    modDict  -- globals()

    Optional parameters:
    package -- Package of the modules (default None signals to use the current package)
    transferModuleName -- Dictionary for names of the device->CPU modules to be loaded and inserted in modDict (if the dictionary does not contain a key 'prefix', a default value of 'prefix+'From<device>' will be used)
    fallbackModuleName -- Name of the devices+CPU product fallback producer to be loaded (default None means prefix+'Fallback')

    Returns a pair of
    - something which looks like an EDProducer picking the product from devices+CPU
    - Task containing all the added modules
    """
    path = ""
    if package is None:
        pkgs = __name__.split(".")
        if len(pkgs) > 1:
            path = ".".join(pkgs[:-1])+"."
    else:
        path = package+"."

    # Per-device producers
    for dt in deviceTypes:
        modName = prefix+dt
        pkg = importlib.import_module(path+modName+"_cfi")
        mod = getattr(pkg, modName)
        modDict[modName] = mod

    # device->CPU
    for dt in deviceTypes:
        if dt == "CPU":
            continue
        transferModName = transferModuleNames.get(dt, prefix+"From"+dt)

        transferModPath = path+transferModName+"_cfi"
        transferModPkg = importlib.import_module(transferModPath)
        transferMod = getattr(transferModPkg, transferModName).clone(src=prefix+dt)
        modDict[transferModName] = transferMod

    # Fallback
    if fallbackModuleName is None:
        fallbackModName = prefix+"Fallback"
    else:
        fallbackModName = fallbackModuleName
    fallbackModPath = path+fallbackModName+"_cfi"
    fallbackModPkg = importlib.import_module(fallbackModPath)
    def _from(s):
        if s == "CPU":
            return s
        return "From"+s
    fallback = getattr(fallbackModPkg, fallbackModName).clone(src=[prefix+_from(dt) for dt in deviceTypes])

    # Paths
    tmp = {}
    for dt in deviceTypes:
        tmp[dt] = cms.Path()

    for dt in deviceTypes:
        p = cms.Path()

        # Add inverted filters until the current device type is found, then insert filter and stop
        # For [CUDA, FPGA, CPU] results in
        # CUDA: CUDAFilter
        # FPGA: ~CUDAFilter + FPGAFilter
        # CPU: ~CUDAFilter + ~FPGAFilter
        for dt2 in deviceTypes:
            if dt2 == "CPU":
                continue
            filt = deviceFilters[dt2]
            if dt2 == dt:
                p += filt
                break
            else:
                p += ~filt

        # Finally add the producer of the type
        p += modDict[prefix+dt]

        # Add (until we get the proper fallback mechanism) the transfer module to the path
        if dt != "CPU":
            transferModName = transferModuleNames.get(dt, prefix+"From"+dt)
            p += modDict[transferModName]

        modDict[prefix+"Path"+dt] = p

    # Task
    task = cms.Task(fallback)

    return (fallback, task)

def setupCUDA(prefix, deviceFilter, modDict,
              package=None, transferModule=None, **kwargs):
    transfer = {}
    if transferModule is not None:
        transfer["CUDA"] = transferModule
    return setupHeterogeneous(prefix, ["CUDA", "CPU"], {"CUDA": deviceFilter}, modDict,
                              package, transfer, **kwargs)
