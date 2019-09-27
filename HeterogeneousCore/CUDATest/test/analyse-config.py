#!/usr/bin/env python3

import sys
import json

class Op:
    def __init__(self, name):
        self._name = name
        self._total = 0
        self._count = 0
        self._isTime = name in ["cpu", "kernel"]

    def add(self, d):
        if self._isTime:
            self._total += d["time"] / 1000.
        else:
            if not "bytes" in d:
                print(d)
            self._total += d["bytes"]
        self._count += 1

    def __str__(self):
        avg = float(self._total) / self._count
        unit = "us" if self._isTime else "bytes"
        return "%s %f %s (%d)" % (self._name, avg, unit, self._count)

def analyseModule(data):
    ops = []
    for op in data[0]:
        ops.append(Op(op["name"]))

    for event in data:
        for i, op in enumerate(event):
            ops[i].add(op)

    for op in ops:
        print(" "+str(op))

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
