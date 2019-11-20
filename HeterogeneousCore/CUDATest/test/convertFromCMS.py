#!/usr/bin/env python3

import re
import json
import copy
import argparse
import collections

def parsePaths(f):
    paths = []
    outputPaths = []
    inPaths = False
    inEndPaths = False

    # Eat unnecessary lines
    line = f.readline()
    while line:
        if "paths:" in line:
            break
        line = f.readline()
    
    # Read paths
    pos = f.tell()
    line = f.readline()
    while line:
        if line[:2] == "  ":
            if inEndPaths and "output" in line:
                outputPaths.append(line.strip())
            else:
                paths.append(line.strip())
        elif "end paths:" in line:
            inEndPaths = True
        else:
            f.seek(pos)
            break
        pos = f.tell()
        line = f.readline()
    return (paths, outputPaths)

def parseModulesOnPaths(f, paths, outputPaths):
    modulesOnPaths = []
    outputModules = []

    re_path = re.compile("modules on (end )?path (?P<path>.*?):")
    
    pos = f.tell()
    line = f.readline()
    includeModules = False
    includeOutputModules = False
    while line:
        m = re_path.search(line)
        if m:
            if m.group("path") in paths:
                includeModules = True
            else:
                includeModules = False
            if m.group("path") in outputPaths:
                includeOutputModules = True
            else:
                includeOutputModules = False
        elif line[:2] == "  ":
            if includeModules:
                modulesOnPaths.append(line.strip())
            elif includeOutputModules:
                outputModules.append(line.strip())
        else:
            f.seek(pos)
            break

        pos = f.tell()
        line = f.readline()

    return (modulesOnPaths, outputModules)

def parseConsumes(f):
    consumes = {}

    line = f.readline()
    if "All modules and modules in the current process whose products they consume:" not in line:
        raise Exception("Unexpected line '%s'" % line.rstrip())
    line = f.readline()
    if "This does not include modules from previous processes or the source" not in line:
        raise Exception("Unexpected line '%s'" % line.rstrip())

    mod_re = re.compile(" (?P<type>\S+)/'(?P<label>.+)'")

    currentModule = None
    line = f.readline()
    while line:
        m = mod_re.search(line)
        if line[:4] == "    ":
            #print("%s -> %s" % (currentModule, m.group("label")))
            consumes[currentModule].append(m.group("label"))
        elif line[:2] == "  ":
            currentModule = m.group("label")
            consumes[currentModule] = []
            #print(currentModule)
        else:
            #print(line.rstrip())
            break
        line = f.readline()

    return consumes

def parseModules(f):
    indexToModule = {}

    # Eat the header
    line = f.readline()
    while line:
        if "Module ID  Module label" in line:
            f.readline() # eat next ---
            break
        line = f.readline()


    mod_re = re.compile("#M (?P<index>\d+)\s+(?P<label>\S+)")

    line = f.readline()
    while line:
        m = mod_re.search(line)
        if m:
            indexToModule[int(m.group("index"))] = m.group("label")
        else:
            break
        
        line = f.readline()

    return indexToModule

def parseModuleTimes(f):
    acquireTimes = collections.defaultdict(list)
    moduleTimes = collections.defaultdict(list)

    # preModuleEventAcquire         A   <Stream ID> <Module ID> <Time since beginJob (us)>
    # postModuleEventAcquire        a   <Stream ID> <Module ID> <Time since beginJob (us)>
    streamModuleFmt = " (?P<stream>\d+) (?P<module>\d+)"
    timeFmt =  " (?P<time>\d+)"
    pre_acquire_re = re.compile("A"+streamModuleFmt+timeFmt)
    post_acquire_re = re.compile("a"+streamModuleFmt+timeFmt)

    # preModuleTransition           M   <Stream ID> <Module ID> <Transition type> <Time since beginJob (us)>
    # postModuleTransition          m   <Stream ID> <Module ID> <Transition type> <Time since beginJob (us)>
    # event transition is 0
    pre_module_re = re.compile("M"+streamModuleFmt+" 0"+timeFmt)
    post_module_re = re.compile("m"+streamModuleFmt+" 0"+timeFmt)

    streamModuleBegin = collections.defaultdict(lambda: collections.defaultdict(int))

    for line in f:
        m = pre_acquire_re.search(line)
        if m:
            modId = int(m.group("module"))
            stream = streamModuleBegin[int(m.group("stream"))]
            if modId in stream:
                raise Exception("Inconsistency")
            stream[modId] = int(m.group("time"))
            continue
        m = post_acquire_re.search(line)
        if m:
            modId = int(m.group("module"))
            stream = streamModuleBegin[int(m.group("stream"))]
            acquireTimes[modId].append(int(m.group("time")) - stream[modId])
            del stream[modId]

        m = pre_module_re.search(line)
        if m:
            modId = int(m.group("module"))
            stream = streamModuleBegin[int(m.group("stream"))]
            if modId in stream:
                raise Exception("Inconsistency")
            stream[modId] = int(m.group("time"))
            continue
        m = post_module_re.search(line)
        if m:
            modId = int(m.group("module"))
            stream = streamModuleBegin[int(m.group("stream"))]
            moduleTimes[modId].append(int(m.group("time")) - stream[modId])
            del stream[modId]

    return (acquireTimes, moduleTimes)

def main(opts):
    with open(opts.tracer) as f:
        (paths, outputPaths) = parsePaths(f)
        #print(paths)
        #print(outputPaths)
        (modules, outputModules) = parseModulesOnPaths(f, paths, outputPaths)
        #print(modules)
        #print(outputModules)
        consumes = parseConsumes(f)
        #print(consumes)


    ignoreLabels = set(paths)
    ignoreLabels.update(outputPaths)
    ignoreLabels.update(outputModules)
    ignoreLabels.add("TriggerResults")

    with open(opts.stallMonitor) as f:
        indexToModule = parseModules(f)
        (acquireTimesRaw, moduleTimesRaw) = parseModuleTimes(f)
        #print(moduleTimes)
        #for m, t in moduleTimes.items():
        #    print("%s: %s" % (indexToModule[m], ", ".join(str(x) for x in t)))
        acquireTimes = {indexToModule[m]: t for m,t in acquireTimesRaw.items()}
        moduleTimes = {indexToModule[m]: t for m,t in moduleTimesRaw.items()}

    #parseTracer(opts.tracer)


    # clean up all modules with @
    def cleanWithAt(d):
        caseLabels = [l for l in d.keys() if "@" in l]
        #print(caseLabels)
        for l in caseLabels:
            bl = l.split("@")[0]
            d[bl] = d[l]
            del d[l]
    cleanWithAt(consumes)
    cleanWithAt(acquireTimes)
    cleanWithAt(moduleTimes)

    # end clean up

    config = {}

    config["moduleConsumes"] = copy.copy(consumes)
    for il in ignoreLabels:
        del config["moduleConsumes"][il]
    out = set(modules)
    for om in outputModules:
        out.update(consumes[om])
    moduleLabels = list(sorted(config["moduleConsumes"].keys()))
    config["moduleConsumes"]["_out"] = list(sorted(out))

    config["moduleSequence"] = moduleLabels[:]

    config["moduleDeclarations"] = {l: "SimCPUEW" if l in acquireTimes else "SimCPU" for l in moduleLabels}
    config["moduleDefinitions"] = {}
    for l in moduleLabels:
        d = {}
        if l in acquireTimes:
            d["acquire"] = [dict(
                name = "cpu",
                values = [int(x*1000) for x in acquireTimes[l]],
                unit = "ns"
            )]
        if l in moduleTimes:
            d["produce"] = [dict(
                name = "cpu",
                values = [int(x*1000) for x in moduleTimes[l]],
                unit = "ns"
            )]
            config["moduleDefinitions"][l] = d
        else:
            # remove all modules that are not run per event, mainly MEtoEDMConverter
            config["moduleSequence"].remove(l)
            del config["moduleConsumes"][l]

    with open(opts.output, "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert files from CMSSW")
    # add
    #if False:
    #    process.add_(cms.Service("StallMonitor", fileName = cms.untracked.string("stallMonitor_reco.log")))
    #    process.maxEvents.input = 50
    #else:
    #    process.Tracer = cms.Service("Tracer", dumpPathsAndConsumes = cms.untracked.bool(True))
    #process.maxEvents.input = 1
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file")
    parser.add_argument("-s", "--stallMonitor", type=str, required=True,
                        help="StallMonitor log file")
    parser.add_argument("-t", "--tracer", type=str, required=True,
                        help="Tracer log file. Tracer service needs 'dumpPathsAndConsumes = cms.untracked.bool(True)'")

    opts = parser.parse_args()

    main(opts)
