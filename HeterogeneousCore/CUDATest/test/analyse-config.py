#!/usr/bin/env python3

import sys
import json
import statistics

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

def main(fname):
    data = None
    with open(fname) as f:
        data = json.load(f)

    modules = data["moduleDefinitions"]
    labels = sorted(modules.keys())
    for label in labels:
        module = modules[label]
        if "acquire" in module:
            print(label+"::acquire()")
            analyseModule(module["acquire"])
        print(label+"::produce()")
        analyseModule(module["produce"])
        print()

if __name__ == "__main__":
    main(sys.argv[1])
