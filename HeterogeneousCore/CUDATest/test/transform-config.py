#!/usr/bin/env python3

import sys
import json
import argparse
import operator
import itertools
import statistics

def functionMean(data):
    for op in data:
        op["values"] = [int(statistics.mean(op["values"]))]
        if "apiTime" in op:
            op["apiTime"] = [int(statistics.mean(op["apiTime"]))]
    return data

def functionMultiplyKernel(data, factor):
    for op in data:
        if op["name"] == "kernel":
            op["values"] = [int(x*factor) for x in op["values"]]
    return data

def functionMultiplyCopy(data, factor):
    for op in data:
        if "memcpy" in op["name"]:
            op["values"] = [int(x*factor) for x in op["values"]]
    return data


def functionCollapse(data):
    ops = []
    for op in data:
        oldop = [x for x in ops if x["name"] == op["name"]]
        if len(oldop) == 0:
            ops.append(op)
        else:
            if len(oldop) != 1:
                raise Exception("LogicError")
            oldop = oldop[0]
            oldop["values"] = list(map(operator.add, oldop["values"], op["values"]))
            if "apiTime" in oldop:
                oldop["apiTime"] = list(map(operator.add, oldop["apiTime"], op["apiTime"]))
            if "function" in oldop:
                oldop["function"] = "(collapsed)"
    return ops

def functionKernelsToCPU(data):
    for op in data:
        if op["name"] == "kernel":
            op["name"] = "cpu"
    return data

def functionCPUToKernel(data):
    for op in data:
        if op["name"] == "cpu":
            op["name"] = "kernel"
    return data

def functionCPUToSleep(data):
    for op in data:
        if op["name"] == "cpu":
            op["name"] = "sleep"
    return data

def functionDropMemcpy(data):
    return [op for op in data if not "memcpy" in op["name"]]

def functionDropMemset(data):
    return [op for op in data if not "memset" in op["name"]]

def functionFakeCUDA(data):
    for op in data:
        if op["name"] == "kernel" or "memcpy" in op["name"] or "memset" in op["name"]:
            op["name"] = "fake"
            op["values"] = op["apiTime"]
            op["unit"] = "ns"
    return data

def filterAll(label, module):
    return True

def filterByGPU(label, module):
    for op in module["produce"]:
        if op["name"] != "cpu":
            return True
    return False

def filterByName(label, module, labels):
    return label in labels

def transformModulePerFunction(label, module, declarations, func, filterFunc=filterAll):
    if filterFunc(label, module):
        if "acquire" in module:
            module["acquire"] = func(module["acquire"])
        module["produce"] = func(module["produce"])

def transformModuleExternalWork(label, module, declarations, filterFunc):
    if "acquire" in module:
        # Already ExternalWork
        return
    if filterFunc(label, module):
        print("Made %s ExternalWork" % label)
        declarations[label] = "SimEW"
        module["acquire"] = module["produce"]
        module["produce"] = []

def transformModuleExternalWorkToSleep(label, module, declarations, filterFunc):
    if filterFunc(label, module):
        if not "acquire" in module:
            raise Exception("Expected module %s to be ExternalWork, but it is not" % label)
        print("Made %s ExternalWork sleeping" % label)
        declarations[label] = "SimEWSleeping"
    

def main(opts):
    transformModules = []

    def appendTransform(tfFunc, *args):
        transformModules.append(lambda l, m, d: tfFunc(l, m, d, *args))

    if len(opts.externalWork) > 0:
        ffunc = None
        if opts.externalWork[0] == "_gpu":
            ffunc = filterByGPU
        elif opts.externalWork[0] == "_all":
            ffunc = filterAll
        else:
            ffunc = lambda l,m: filterByName(l, m, opts.externalWork)
        appendTransform(transformModuleExternalWork, ffunc)
    if opts.mean:
        appendTransform(transformModulePerFunction, functionMean)
    if opts.multiplyKernel is not None:
        appendTransform(transformModulePerFunction, lambda da: functionMultiplyKernel(da, opts.multiplyKernel))
    if opts.multiplyCopy is not None:
        appendTransform(transformModulePerFunction, lambda da: functionMultiplyCopy(da, opts.multiplyCopy))
    if opts.kernelsToCPU:
        appendTransform(transformModulePerFunction, functionKernelsToCPU)
    if len(opts.cpuToKernel) > 0:
        ffunc = lambda l,m: filterByName(l, m, opts.cpuToKernel)
        appendTransform(transformModulePerFunction, functionCPUToKernel, ffunc)
    if len(opts.cpuToSleep) > 0:
        ffunc = lambda l,m: filterByName(l, m, opts.cpuToSleep)
        appendTransform(transformModuleExternalWorkToSleep, ffunc)
        appendTransform(transformModulePerFunction, functionCPUToSleep, ffunc)
    if opts.dropMemcpy:
        appendTransform(transformModulePerFunction, functionDropMemcpy)
    if opts.dropMemset:
        appendTransform(transformModulePerFunction, functionDropMemset)
    if opts.fakeCUDA:
        appendTransform(transformModulePerFunction, functionFakeCUDA)
    if opts.collapse:
        appendTransform(transformModulePerFunction, functionCollapse)

    if len(transformModules) == 0 and opts.merge is None:
        raise Exception("No transform operation were given")

    data = None
    with open(opts.file) as f:
        data = json.load(f)

    if opts.merge is not None:
        with open(opts.merge) as f:
            data2 = json.load(f)
        data["moduleDeclarations"].extend(data2["moduleDeclarations"])
        for label, module in data2["moduleDefinitions"].items():
            if label in data["moduleDefinitions"]:
                raise Exception("Module %s already found from input file, unable to merge" % label)
            data["moduleDefinitions"][label] = module

    declarations = data.get("moduleDeclarations", {})

    for label, module in data["moduleDefinitions"].items():
        for func in transformModules:
            func(label, module, declarations)

    with open(opts.output, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform config JSON file")
    parser.add_argument("file", type=str,
                        help="Input JSON file")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file")
    parser.add_argument("--merge", type=str, default=None,
                        help="Merge this file with the input file to output")
    parser.add_argument("--mean", action="store_true",
                        help="Replace each operation event-by-event values with the mean")
    parser.add_argument("--externalWork", type=str, default=None,
                        help="Comma-separated list of modules who are changed to ExternalWork. Special values: _gpu for all GPU modules, and _all for all modules.")
    parser.add_argument("--collapse", action="store_true",
                        help="Collapse all same-kind-of operations to one per module function")
    parser.add_argument("--kernelsToCPU", action="store_true",
                        help="Change all GPU kernels to CPU work with the same timing")
    parser.add_argument("--cpuToKernel", type=str, default=None,
                        help="Comma-separated list of modules whose CPU time is changed to kernel")
    parser.add_argument("--cpuToSleep", type=str, default=None,
                        help="Comma-separated list of modules whose CPU time is changed to sleep.")
    parser.add_argument("--dropMemcpy", action="store_true",
                        help="Drop all memcopies")
    parser.add_argument("--dropMemset", action="store_true",
                        help="Drop all memsets")
    parser.add_argument("--fakeCUDA", action="store_true",
                        help="Fake CUDA operations by burning CPU for the duration of API call")
    parser.add_argument("--multiplyKernel", type=float, default=None,
                        help="Multiply all kernel lengths with this value (default: None)")
    parser.add_argument("--multiplyCopy", type=float, default=None,
                        help="Multiply all memcpy lengths with this value (default: None)")

    opts = parser.parse_args()

    if opts.externalWork is not None:
        opts.externalWork = opts.externalWork.split(",")
        if "_gpu" in opts.externalWork:
            if len(opts.externalWork) != 1:
                parser.error("Got _gpu for --externalWork, but also other items")
        if "_all" in opts.externalWork:
            if len(opts.externalWork) != 1:
                parser.error("Got _all for --externalWork, but also other items")
    else:
        opts.externalWork = []
    if opts.cpuToKernel is not None:
        opts.cpuToKernel = opts.cpuToKernel.split(",")
    else:
        opts.cpuToKernel = []
    if opts.cpuToSleep is not None:
        opts.cpuToSleep = opts.cpuToSleep.split(",")
    else:
        opts.cpuToSleep = []

    main(opts)
