#!/usr/bin/env python3

import sys
import json
import statistics

def analyseModule(data):
    for op in data:
        isTime = op["name"] in ["cpu", "kernel"]
        values = op["time"] if isTime else op["bytes"]
        unit = "us" if isTime else "bytes"

        avg = statistics.mean(values)
        if isTime:
            avg = avg/1000.
        stddev = ""
        if len(values) > 1:
            s = statistics.stdev(values)
            if isTime:
                s = s/1000.
            stddev = " sigma %.2f" % s

        metadata = ""
        if "function" in op:
            metadata = " "+op["function"]

        print(" %s %.2f%s %s (%d)%s" % (op["name"], avg, stddev, unit, len(values), metadata))

def main(fname):
    data = None
    with open(fname) as f:
        data = json.load(f)
    
    for label, module in data["moduleDefinitions"].items():
        if "acquire" in module:
            print(label+"::acquire()")
            analyseModule(module["acquire"])
        print(label+"::produce()")
        analyseModule(module["produce"])
        print()

if __name__ == "__main__":
    main(sys.argv[1])
