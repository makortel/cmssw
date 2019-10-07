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

def functionDropMemcpy(data):
    return [op for op in data if not "memcpy" in op["name"]]

def functionDropMemset(data):
    return [op for op in data if not "memset" in op["name"]]

def transformModulePerFunction(label, module, func):
    if "acquire" in module:
        module["acquire"] = func(module["acquire"])
    module["produce"] = func(module["produce"])

def transformModuleExternalWork(label, module):
    if "acquire" in module:
        # Already ExternalWork
        return
    hasGPUops = False
    for op in module["produce"]:
        if op["name"] != "cpu":
            hasGPUops = True
            break
    if hasGPUops:
        print("Made %s ExternalWork" % label)
        module["acquire"] = module["produce"]
        module["produce"] = []

def main(opts):
    transformModules = []
    if opts.externalWork:
        transformModules.append(transformModuleExternalWork)
    if opts.mean:
        transformModules.append(lambda l, m: transformModulePerFunction(l, m, functionMean))
    if opts.kernelsToCPU:
        transformModules.append(lambda l, m: transformModulePerFunction(l, m, functionKernelsToCPU))
    if opts.dropMemcpy:
        transformModules.append(lambda l, m: transformModulePerFunction(l, m, functionDropMemcpy))
    if opts.dropMemset:
        transformModules.append(lambda l, m: transformModulePerFunction(l, m, functionDropMemset))
    if opts.collapse:
        transformModules.append(lambda l, m: transformModulePerFunction(l, m, functionCollapse))

    if len(transformModules) == 0:
        raise Exception("No transform operation were given")

    data = None
    with open(opts.file) as f:
        data = json.load(f)
    
    for label, module in data["moduleDefinitions"].items():
        for func in transformModules:
            func(label, module)

    with open(opts.output, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform config JSON file")
    parser.add_argument("file", type=str,
                        help="Input JSON file")
    parser.add_argument("-o", "--output", type=str,
                        help="Output file")
    parser.add_argument("--mean", action="store_true",
                        help="Replace each operation event-by-event values with the mean")
    parser.add_argument("--externalWork", action="store_true",
                        help="Make all GPU modules ExteralWork")
    parser.add_argument("--collapse", action="store_true",
                        help="Collapse all same-kind-of operations to one per module function")
    parser.add_argument("--kernelsToCPU", action="store_true",
                        help="Change all GPU kernels to CPU work with the same timing")
    parser.add_argument("--dropMemcpy", action="store_true",
                        help="Drop all memcopies")
    parser.add_argument("--dropMemset", action="store_true",
                        help="Drop all memsets")

    opts = parser.parse_args()

    main(opts)
