#!/usr/bin/env python3

import os
import sys
import copy
import json
import ctypes
import sqlite3
import argparse
import subprocess

# inspired and partly copied from https://github.com/ezyang/nvprof2json/blob/master/nvprof2json.py

class Marker:
    def __init__(self, id, name, domain, color, start, processId, threadId):
        self._id = id
        self._name = name
        self._domain = domain
        self._color = color
        self._start = start
        self._stop = None
        self._processId = processId
        self._threadId = threadId

    def setStop(self, stop):
        self._stop = stop
        
    def name(self):
        return self._name

    def processId(self):
        return self._processId

    def threadId(self):
        return self._threadId

    def duration(self):
        return self._stop - self._start

class Memcpy:
    def __init__(self, copyKind, srcKind, dstKind, bytes, start, stop, correlationId, apiStart, apiStop, threadId):
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
        # CUPTI_ACTIVITY_KIND_RUNTIME.threadId stores unsigned int in signed field...
        self._threadId = ctypes.c_uint(threadId).value

    def __str__(self):
        #return "memcpy %s %d bytes %f us" % (self._copyKind, self._bytes, self.apiDuration()/1000.)
        return "memcpy %s %d bytes" % (self._copyKind, self._bytes)

    def toOp(self):
        return Op("memcpy"+self._copyKind, self._bytes, unit="bytes", apiTime=self.apiDuration())
    
    def duration(self):
        return self._stop - self._start

    def apiDuration(self):
        return self._apiStop - self._apiStart

    def threadId(self):
        return self._threadId

class Memset:
    def __init__(self, bytes, start, stop, correlationId, apiStart, apiStop, threadId):
        self._bytes = bytes
        self._start = start
        self._stop = stop
        self._correlationId = correlationId
        self._apiStart = apiStart
        self._apiStop = apiStop
        # CUPTI_ACTIVITY_KIND_RUNTIME.threadId stores unsigned int in signed field...
        self._threadId = ctypes.c_uint(threadId).value

    def __str__(self):
        #return "memset %s %d bytes %f us" % (self._copyKind, self._bytes, self.apiDuration()/1000.)
        return "memset %d bytes" % (self._bytes)

    def toOp(self):
        return Op("memset", self._bytes, unit="bytes", apiTime=self.apiDuration())
    
    def duration(self):
        return self._stop - self._start

    def apiDuration(self):
        return self._apiStop - self._apiStart

    def threadId(self):
        return self._threadId

class Kernel:
    def __init__(self, start, stop, name, correlationId, apiStart, apiStop, threadId):
        self._start = start
        self._stop = stop
        self._name = name
        self._shortName = name[0:name.find("(", 1)] # if the demangled name starts with (, ignore that match (to support anonymous namespace)
        self._correlationId = correlationId
        self._apiStart = apiStart
        self._apiStop = apiStop
        # CUPTI_ACTIVITY_KIND_RUNTIME.threadId stores unsigned int in signed field...
        self._threadId = ctypes.c_uint(threadId).value

    def __str__(self):
        #return "kernel %f us %f us %s" % (self.duration()/1000., self.apiDuration()/1000., self._name)
        return "kernel %f us %s" % (self.duration()/1000., self._shortName)

    def toOp(self):
        return Op("kernel", self.duration(), unit="ns", apiTime=self.apiDuration(), func=self._shortName)
        
    def duration(self):
        return self._stop - self._start

    def apiDuration(self):
        return self._apiStop - self._apiStart

    def threadId(self):
        return self._threadId

class Op:
    def __init__(self, name, value, unit, apiTime=None, func=None):
        self._name = name
        self._values = [value]
        self._func = func
        self._apiTime = [apiTime] if apiTime is not None else None
        self._unit = unit

    def isSame(self, op):
        return self._name == op._name and self._func == op._func

    def add(self, op):
        self._values.extend(op._values)
        if self._apiTime is not None:
            self._apiTime.extend(op._apiTime)

    def nevents(self):
        return len(self._values)

    def __str__(self):
        if self._func is None:
            return "%s %d" % (self._name, self._values[0])
        return "%s(%s) %d" % (self._name, self._func, self._values[0])

    def toDict(self):
        ret = dict(
            name = self._name,
            values = self._values,
            unit = self._unit
        )
        if self._apiTime is not None:
            ret["apiTime"] = self._apiTime
        if self._func is not None:
            ret["function"] = self._func
        return ret
   
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
        self._addModules(self._acquire, ops)

    def addProduces(self, ops):
        self._addModules(self._produce, ops)

    def _addModules(self, lst, ops):
        if len(lst) == 0:
            lst.extend(ops)
            return
        iop = 0
        for oin in ops:
            o = lst[iop]
            if o.isSame(oin):
                o.add(oin)
            else:
                raise Exception("Op %s not found in module %s" % (str(oin), self._name))
            iop += 1

    def declaration(self):
        cpptype = "ProducerExternalWork" if self.hasAcquire() else "Producer"
        return "process.%s = cms.EDProducer('%s')" % (self._name, cpptype)

    def toDict(self):
        ret = {}
        nevents = self._produce[0].nevents()
        if self.hasAcquire():
            acquire = []
            for op in self._acquire:
                if op.nevents() != nevents:
                    raise Exception("Module %s got inconsistent amount of events, produce[0] has %d, op '%s' in acquire() has %d" % (self._name, nevents, str(op), op.nevents))
                acquire.append(op.toDict())
            ret["acquire"] = acquire
        produce = []
        for op in self._produce:
            if op.nevents() != nevents:
                raise Exception("Module %s got inconsistent amount of events, produce[0] has %d, op '%s' in produce() has %d" % (self._name, nevents, str(op), op.nevents))
            produce.append(op.toDict())
        ret["produce"] = produce
        return ret

def main(opts):
    conn = sqlite3.connect(opts.file)
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r[0]] = demangle(r[1])

    #for i, name in strings.items():
    #    print(i, name)


    # Find information on markers
    markersDict = {}
    modules = set()
    threadIds = set()
    for r in conn.execute("SELECT CUPTI_ACTIVITY_KIND_MARKER.id, name, domain, timestamp, color, objectKind, objectId FROM CUPTI_ACTIVITY_KIND_MARKER INNER JOIN CUPTI_ACTIVITY_KIND_MARKER_DATA on CUPTI_ACTIVITY_KIND_MARKER_DATA.id = CUPTI_ACTIVITY_KIND_MARKER.id"):
        (mid, name, domain, timestamp, color, objectKind, objectId) = r
        name = strings[name]

        if mid in markersDict:
            markersDict[mid].setStop(timestamp)
        else:
            # if not an (process, thread ID), ignore marker
            if objectKind != 2:
                continue
            # Ok, this is now a very ugly hack to interpret union{struct{uint32, uint32}, struct{uint32, uint32, uint32}}
            # Will definitively NOT work in general
            if len(objectId) != 12:
                raise Exception("Got an objectID with size of %d bytes instead of 12" % len(objectId))
            processId = int.from_bytes((objectId[0:4]), byteorder='little')
            threadId = int.from_bytes((objectId[4:8]), byteorder='little')
            empty = int.from_bytes((objectId[8:12]), byteorder='little')
            if empty != 0:
                raise Exception("Something is wrong, got %d instead of 0" % empty)
            threadIds.add(threadId)

            markersDict[mid] = Marker(mid, name, domain, color, timestamp, processId, threadId)
            if " construction" in name:
                modules.add(name.split(" ")[0])

    #print("\n".join(modules))

    # Ignore some "modules"
    for x in ["raw2digi_step", "reconstruction_step", "TriggerResults", "outPath", "out", "TVreco", "Raw2Hit", "toSoA"]:
        try:
            modules.remove(x)
        except KeyError: pass

    #print("\n".join(modules))
    
    # Keep only markers relating to the modules, sort them by time
    markers = list(filter(lambda x: x.name() in modules or ("acquire" in x.name() and x.name().split(" ")[0] in modules), markersDict.values()))
    markers.sort(key = lambda x: x._start)
    # Skip first event(s)
    if opts.skipEvents > 0:
        pos = None
        nskip = opts.skipEvents+1
        for i, m in enumerate(markers):
            if m.name() == "source":
                pos = i
                nskip -= 1
                if nskip == 0:
                    break
        # # pos holds now the index of the first "source" to be kept
        markers = markers[pos:]

        # This logic is not really needed?
        # That's right, with single EDM stream the "source" acts as a barrier (no modules from previous event after source, no modules from next event before source)
        # With multiple EDM streams something like the following could become helpful, but let's not go there yet
        # # pos holds now the index of the first "source" to be kept
        # # its thread is therefore fine
        # threadIds_skip = copy.copy(threadIds)
        # threadIds_skip.remove(m.threadId())
        # # for the remaining threads, skip markers until a source is reached
        # markers_new = []
        # pos_new = None
        # for i, m in enumerate(markers[pos:]):
        #     if m.threadId() in threadIds_skip:
        #         if m.name() == "source":
        #             threadIds_skip.remove(m.threadId())
        #             markers_new.append(m)
        #             if len(threadIds_skip) == 0:
        #                 pos_new = i+1
        #                 break
        #         # else, skip marker
        #     else:
        #         markers_new.append(m)
        # 
        # if pos_new is not None:
        #     markers_new.extend(markers[pos_new:])
        # markers = markers_new

    #for m in markers[:20]:
    #    if m._stop is not None:
    #        print(m._threadId, m.name(), m.duration(), m._domain, m._color)

    # Find memcpy's
    memcpy = []
    for r in conn.execute("SELECT copyKind, srcKind, dstKind, bytes, CUPTI_ACTIVITY_KIND_MEMCPY.start, CUPTI_ACTIVITY_KIND_MEMCPY.end, CUPTI_ACTIVITY_KIND_MEMCPY.correlationId, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end, CUPTI_ACTIVITY_KIND_RUNTIME.threadId FROM CUPTI_ACTIVITY_KIND_MEMCPY INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_MEMCPY.correlationId ORDER BY CUPTI_ACTIVITY_KIND_RUNTIME.start"):
        memcpy.append(Memcpy(*r))

    # Find memset's
    memset = []
    for r in conn.execute("SELECT bytes, CUPTI_ACTIVITY_KIND_MEMSET.start, CUPTI_ACTIVITY_KIND_MEMSET.end, CUPTI_ACTIVITY_KIND_MEMSET.correlationId, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end, CUPTI_ACTIVITY_KIND_RUNTIME.threadId FROM CUPTI_ACTIVITY_KIND_MEMSET INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_MEMSET.correlationId WHERE CUPTI_ACTIVITY_KIND_MEMSET.memoryKind = 3 ORDER BY CUPTI_ACTIVITY_KIND_RUNTIME.start"):
        memset.append(Memset(*r))

    # Correlation ID can be used to JOIN with API calls in CUPTI_ACTIVITY_KIND_RUNTIME
    # cbid value are from /usr/local/cuda-10.1/extras/CUPTI/include/cupti_runtime_cbid.h

    # Find kernels
    kernels = []
    for r in conn.execute("SELECT CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.start, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.end, name, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.correlationId, CUPTI_ACTIVITY_KIND_RUNTIME.start, CUPTI_ACTIVITY_KIND_RUNTIME.end, CUPTI_ACTIVITY_KIND_RUNTIME.threadId FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME on CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.correlationId ORDER BY CUPTI_ACTIVITY_KIND_RUNTIME.start"):
        kernels.append(Kernel(r[0], r[1], strings[r[2]], r[3], r[4], r[5], r[6]))

    maxMod = opts.maxModules
    maxEv = opts.maxEvents
    mods = {x: Module(x) for x in modules}
    iEv = -1
    for m in markers:
        #print(m.name(), m.threadId())
        if m.name() == "source":
            iEv += 1
            continue
        if maxEv >= 0:
            if iEv == maxEv:
                break

        if m._stop is None:
            print("dropping marker '%s' because of missing stop time" % m.name())
            continue
        ops = []
        ops_time = 0
        # match operations to markers by thread id and time stamp
        for opslist in [memcpy, memset, kernels]:
            ops_tmp = []
            for c in opslist:
                if c.threadId() != m.threadId():
                    continue
                if c._apiStart >= m._start and c._apiStop <= m._stop:
                    ops_tmp.append(c)
                    ops_time += c.apiDuration()
                if c._apiStop > m._stop:
                    break
            # remove gathered ops, can not be assigned to another marker
            for o in ops_tmp:
                opslist.remove(o)
            ops.extend(ops_tmp)
        ops.sort(key = lambda x: x._apiStart)

        cpuTime = m.duration()-ops_time
        #print("%s time %f us, ops time %f us, diff %f us" % (m.name(), m.duration()/1000., ops_time/1000., cpuTime/1000.))
        #for op in ops:
        #    print(" %s" % str(op))
        #print(m.name())
        try:
            if "acquire" in m.name():
                mods[m.name().split(" ")[0]].addAcquires([Op("cpu", cpuTime, unit="ns")] +
                                                         [x.toOp() for x in ops])
            else:
                mods[m.name()].addProduces([Op("cpu", cpuTime, unit="ns")] +
                                           [x.toOp() for x in ops])
        except:
            print("Event %d" % iEv)
            print("%s time %f us, ops time %f us, diff %f us" % (m.name(), m.duration()/1000., ops_time/1000., cpuTime/1000.))
            for op in ops:
                print(" %s" % str(op))
            raise

        if maxMod >= 0:
            maxMod -= 1
            if maxMod == 0:
                break

    #mods["source"].rename("sourceNew")
    if "source" in mods:
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
    parser.add_argument("--skipEvents", type=int, default=1,
                        help="Number of events to be skipped (default: 1)")

    opts = parser.parse_args()

    main(opts)
