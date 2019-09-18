#!/usr/bin/env python3

import os
import sys
import json
import sqlite3
import argparse
import subprocess

# inspired and partly copied from https://github.com/ezyang/nvprof2json/blob/master/nvprof2json.py

class Marker:
    def __init__(self, id, name, domain, color, start):
        self._id = id
        self._name = name
        self._domain = domain
        self._color = color
        self._start = start
        self._stop = None

    def setStop(self, stop):
        self._stop = stop
        
    def name(self):
        return self._name

    def duration(self):
        return self._stop - self._start

class Memcpy:
    def __init__(self, copyKind, srcKind, dstKind, bytes, start, stop, correlationId, apiStart, apiStop):
        self._copyKind = {1: "HtoD", 2: "DtoH"}[copyKind]
        _memKind = {1: "pageable", 2: "pinned", 3: "device"}
        self._srcKind = _memKind[srcKind]
        self._dstKind = _memKind[dstKind]
        self._bytes = bytes
        self._start = start
        self._stop = stop
        self._correlationId = correlationId
        self._apiStart = apiStart
        self._apiStop = apiStop

    def __str__(self):
        #return "memcpy %s %d bytes %f us" % (self._copyKind, self._bytes, self.apiDuration()/1000.)
        return "memcpy %s %d bytes" % (self._copyKind, self._bytes)

    def toOp(self):
        return Op("memcpy"+self._copyKind, self._bytes)
    
    def duration(self):
        return self._stop - self._start

    def apiDuration(self):
        return self._apiStop - self._apiStart
        

class Kernel:
    def __init__(self, start, stop, name, correlationId, apiStart, apiStop):
        self._start = start
        self._stop = stop
        self._name = name
        self._correlationId = correlationId
        self._apiStart = apiStart
        self._apiStop = apiStop

    def __str__(self):
        #return "kernel %f us %f us %s" % (self.duration()/1000., self.apiDuration()/1000., self._name)
        return "kernel %f us %s" % (self.duration()/1000., self._name)

    def toOp(self):
        return Op("kernel", self.duration())
        
    def duration(self):
        return self._stop - self._start

    def apiDuration(self):
        return self._apiStop - self._apiStart

class Op:
    def __init__(self, name, value):
        self._name = name
        self._value = value

    def __str__(self):
        if "memcpy" in self._name:
            return "cms.PSet(name = cms.string('%s'), bytes = cms.uint32(%d))" % (self._name, self._value)
        else:
            return "cms.PSet(name = cms.string('%s'), time = cms.uint64(%d))" % (self._name, self._value)

    def toDict(self):
        if "memcpy" in self._name:
            return dict(name = self._name, bytes = self._value)
        else:
            return dict(name = self._name, time = self._value)
   
class Module:
    def __init__(self, name):
        self._name = name
        self._acquire = []
        self._produce = []

    def name(self):
        return self._name

    def rename(self, name):
        self._name = name

    def hasContent(self):
        return len(self._produce) > 0 or len(self._acquire) > 0
        
    def hasAcquire(self):
        return len(self._acquire) > 0
        
    def addAcquires(self, ops):
        self._acquire.append(ops)

    def extendAcquires(self, ops):
        self._acquire[-1].extend(ops)

    def addProduces(self, ops):
        self._produce.append(ops)

    def extendProduces(self, ops):
        self._produce[-1].extend(ops)

    def declaration(self):
        cpptype = "ProducerExternalWork" if self.hasAcquire() else "Producer"
        return "process.%s = cms.EDProducer('%s')" % (self._name, cpptype)

    def __str__(self):
        ret = ""
        if self.hasAcquire():
            if len(self._acquire) != len(self._produce):
                raise Exception("Module %s got different amount of acquire events (%d) and produce events (%d)" % (self._name, len(self._acquire), len(self._produce)))
            ret += "process.%s.acquire = cms.VPSet(\n" % self._name
            for ev in self._acquire:
                ret += "    cms.PSet(event = cms.VPSet(\n"
                for op in ev:
                    ret += "        "+str(op)+",\n"
                ret += "    ),\n"
            ret += ")\n"
        ret += "process.%s.produce = cms.VPSet(\n" % self._name
        for ev in self._produce:
                ret += "    cms.PSet(event = cms.VPSet(\n"
                for op in ev:
                    ret += "        "+str(op)+",\n"
                ret += "    ),\n"
        ret += ")\n"
        return ret

    def toDict(self):
        ret = {}
        if self.hasAcquire():
            if len(self._acquire) != len(self._produce):
                raise Exception("Module %s got different amount of acquire events (%d) and produce events (%d)" % (self._name, len(self._acquire), len(self._produce)))
            acquire = []
            for ev in self._acquire:
                acquire.append([op.toDict() for op in ev])
            ret["acquire"] = acquire
        produce = []
        for ev in self._produce:
            produce.append([op.toDict() for op in ev])
        ret["produce"] = produce
        return ret

def main(opts):
    conn = sqlite3.connect(opts.file)
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r[0]] = demangle(r[1])

        
    #for i, name in strings.items():
    #    print(i, name)


    markersDict = {}
    modules = set()
    for r in conn.execute("SELECT CUPTI_ACTIVITY_KIND_MARKER.id, name, domain, timestamp, color FROM CUPTI_ACTIVITY_KIND_MARKER INNER JOIN CUPTI_ACTIVITY_KIND_MARKER_DATA on CUPTI_ACTIVITY_KIND_MARKER_DATA.id = CUPTI_ACTIVITY_KIND_MARKER.id"):
        if r[0] in markersDict:
            markersDict[r[0]].setStop(r[3])
        else:
            name = strings[r[1]]
            markersDict[r[0]] = Marker(r[0], name, r[2], r[4], r[3])
            if " construction" in name:
                modules.add(name.split(" ")[0])
    #print("\n".join(modules))

    # Ignore some "modules"
    for x in ["raw2digi_step", "reconstruction_step", "TriggerResults", "outPath", "out", "TVreco", "Raw2Hit"]:
        try:
            modules.remove(x)
        except KeyError: pass
    
            
    markers = list(filter(lambda x: x.name() in modules or ("acquire" in x.name() and x.name().split(" ")[0] in modules), markersDict.values()))
    markers.sort(key = lambda x: x._start)
    # Skip first event
    pos = None
    for i, m in enumerate(markers[1:]):
        if m.name() == "source":
            pos = i+1
            break
    markers = markers[pos:]
        

    #for i in sorted(markersDict.keys()):
    #    m = markersDict[i]
    #    if m._stop is not None:
    #        print(m.name(), m.duration(), m._domain, m._color)

    memcpy = []
    for r in conn.execute("SELECT copyKind, srcKind, dstKind, bytes, CUPTI_ACTIVITY_KIND_MEMCPY.start, CUPTI_ACTIVITY_KIND_MEMCPY.end, CUPTI_ACTIVITY_KIND_MEMCPY.correlationId, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end FROM CUPTI_ACTIVITY_KIND_MEMCPY INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_MEMCPY.correlationId"):
        memcpy.append(Memcpy(*r))
    memcpy.sort(key = lambda x: x._apiStart)

    # Correlation ID can be used to JOIN with API calls in CUPTI_ACTIVITY_KIND_RUNTIME
    # cbid value are from /usr/local/cuda-10.1/extras/CUPTI/include/cupti_runtime_cbid.h

    kernels = []
    for r in conn.execute("SELECT CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.start, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.end, name, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.correlationId, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.correlationId"):
        kernels.append(Kernel(r[0], r[1], strings[r[2]], r[3], r[4], r[5]))
    kernels.sort(key = lambda x: x._apiStart)


    maxMod = opts.maxModules
    maxEv = opts.maxEvents
    mods = {x: Module(x) for x in modules}
    for m in markers:
        if m._stop is None:
            continue
        ops = []
        ops_time = 0
        for c in memcpy:
            if c._apiStart >= m._start and c._apiStop <= m._stop:
                ops.append(c)
                ops_time += c.apiDuration()
            if c._apiStop > m._stop:
                break

        for k in kernels:
            if k._apiStart >= m._start and k._apiStop <= m._stop:
                ops.append(k)
                ops_time += k.apiDuration()
            if k._apiStop > m._stop:
                break
        ops.sort(key = lambda x: x._apiStart)

        cpuTime = m.duration()-ops_time
        #print("%s time %f us, ops time %f us, diff %f us" % (m.name(), m.duration()/1000., ops_time/1000., cpuTime/1000.))
        #for op in ops:
        #    print(" %s" % str(op))
        if "acquire" in m.name():
            mods[m.name().split(" ")[0]].addAcquires([Op("cpu", cpuTime)] +
                                                     [x.toOp() for x in ops])
        else:
            # move all ops from produce to acquire where they really should be in case there was an acquire
            mod = mods[m.name()]
            mod.addProduces([Op("cpu", cpuTime)])
            if len(ops) > 0 and mod.hasAcquire():
                mod.extendAcquires([x.toOp() for x in ops])
            else:
                mod.extendProduces([x.toOp() for x in ops])

        if maxMod >= 0:
            maxMod -= 1
            if maxMod == 0:
                break
        if maxEv >= 0 and m.name() == "source":
            maxEv -= 1
            if maxEv == 0:
                break

    #mods["source"].rename("sourceNew")
    del mods["source"]

    data = dict(
        moduleDeclarations = [mod.declaration() for mod in mods.values() if mod.hasContent()],
        moduleDefinitions = {mod.name(): mod.toDict() for mod in mods.values() if mod.hasContent()}
    )

    with open(opts.output, "w") as out:
        json.dump(data, out, indent=2)
        
def demangle(name):
    """Demangle a C++ identifier using c++filt"""
    # TODO: create the process only once.
    # Fortunately, this doesn't seem to be a bottleneck ATM.
    try:
        with open(os.devnull, 'w') as devnull:
            return subprocess.check_output(['c++filt', '-n', name], stderr=devnull).rstrip().decode("ascii")
    except subprocess.CalledProcessError:
        return name
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exctract module kernels, memory transfers, and additional spent CPU time into a python configuration for CMSSW")
    parser.add_argument("file", type=str,
                        help="nvvp file for input")
    parser.add_argument("-o", "--output", type=str, default="config.json",
                        help="Output file (default: config.json)")
    parser.add_argument("--maxModules", type=int, default=-1,
                        help="Maximum number of modules to process, -1 for all (default: -1")
    parser.add_argument("--maxEvents", type=int, default=10,
                        help="Maximum number of events to process, -1 for all (default: 10")

    opts = parser.parse_args()

    main(opts)
