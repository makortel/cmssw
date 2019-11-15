#!/usr/bin/env python3

import sys
import json
import statistics

class ModuleTime:
    def __init__(self):
        self._cpu = 0
        self._api = 0
        self._kernel = 0

    def cpu(self):
        return self._cpu

    def api(self):
        return self._api

    def kernel(self):
        return self._kernel

    def addCPU(self, t):
        self._cpu += t

    def addAPI(self, t):
        self._api += t

    def addKernel(self, t):
        self._kernel += t

    def add(self, mt):
        self._cpu += mt._cpu
        self._api += mt._api
        self._kernel += mt._kernel

def gatherModule(data):
    t = ModuleTime()
    for op in data:
        avg = statistics.mean(op["values"])
        if op["name"] == "cpu":
            t.addCPU(avg*1e-3)
        if op["name"] == "kernel":
            t.addKernel(avg*1e-3)
        if "apiTime" in op:
            t.addAPI(statistics.mean(op["apiTime"]))
    return t

def analyseModule(data):
    for op in data:
        values = op["values"]
        unit = op["unit"]

        avg = statistics.mean(values)
        if unit == "ns":
            avg = avg*1e-3
            unit = "us"
        stddev = ""
        if len(values) > 1:
            s = statistics.stdev(values)
            if unit == "us":
                s = s*1e-3
            stddev = " sigma %.2f" % s

        apiTime= ""
        if "apiTime" in op:
            times = op["apiTime"]
            apiTime = " in API %.2f" % (statistics.mean(times)*1e-3)
            if len(times) > 1:
                apiTime += " sigma %.2f" % (statistics.stdev(times)*1e-3)
            apiTime += " us"

        metadata = ""
        if "function" in op:
            metadata = " "+op["function"]

        print(" %s %.2f%s %s%s (%d)%s" % (op["name"], avg, stddev, unit, apiTime, len(values), metadata))

def printFunction(label, name, functionTimes, totalTime):
    ti = functionTimes[label+"_"+name]
    print("%s::%s, cpu %.2f us (%.1f %%) api %.2f us (%.1f %%) kernel %.2f us (%.1f %%)" % (label, name,
                                                                                      ti.cpu(), ti.cpu()/totalTime.cpu()*100,
                                                                                      ti.api(), ti.api()/totalTime.api()*100 if totalTime.api() > 0 else 0,
                                                                                      ti.kernel(), ti.kernel()/totalTime.kernel()*100 if totalTime.kernel() > 0 else 0
                                                                                  ))

def main(fname):
    data = None
    with open(fname) as f:
        data = json.load(f)

    modules = data["moduleDefinitions"]
    labels = sorted(modules.keys())
    functionTimes = {}
    totalTime = ModuleTime()
    for label in labels:
        module = modules[label]
        ti = ModuleTime()
        if "acquire" in module:
            ti = gatherModule(module["acquire"])
            totalTime.add(ti)
            functionTimes[label+"_acquire"] = ti
        ti = gatherModule(module["produce"])
        totalTime.add(ti)
        functionTimes[label+"_produce"] = ti

    print("Total CPU time %.2f us, API time %.2f us, kernel time %.2f us" % (totalTime.cpu(), totalTime.api(), totalTime.kernel()))
    print()

    for label in labels:
        module = modules[label]
        if "acquire" in module:
            printFunction(label, "acquire", functionTimes, totalTime)
            analyseModule(module["acquire"])
        printFunction(label, "produce", functionTimes, totalTime)
        analyseModule(module["produce"])
        print()

if __name__ == "__main__":
    main(sys.argv[1])
