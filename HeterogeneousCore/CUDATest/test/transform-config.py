#!/usr/bin/env python3

import sys
import json
import argparse
import operator
import itertools
import statistics

def valueField(op):
    isTime = op["name"] in ["cpu", "kernel"]
    return "time" if isTime else "bytes"

def functionMean(data):
    for op in data:
        vname = valueField(op)
        op[vname] = [int(statistics.mean(op[vname]))]
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
            vname = valueField(op)
            oldop[vname] = list(map(operator.add, oldop[vname], op[vname]))
            if "function" in oldop:
                oldop["function"] = "(collapsed)"
    return ops

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
    transformModule = None
    if opts.mean:
        transformModule = lambda l, m: transformModulePerFunction(l, m, functionMean)
    elif opts.externalWork:
        transformModule = transformModuleExternalWork
    elif opts.collapse:
        transformModule = lambda l, m: transformModulePerFunction(l, m, functionCollapse)
    else:
        raise Exception("No transform operation were given")

    data = None
    with open(opts.file) as f:
        data = json.load(f)
    
    for label, module in data["moduleDefinitions"].items():
        transformModule(label, module)

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

    opts = parser.parse_args()

    main(opts)
