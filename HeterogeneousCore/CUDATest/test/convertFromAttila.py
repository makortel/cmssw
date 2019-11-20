#!/usr/bin/env python3

import json
import networkx as nx
import argparse

def sanitize(n):
    if isinstance(n, str):
        return n.replace("_", "").replace(":", "")
    return [sanitize(x) for x in n]

def fillSequence(seq, seqName, controlFlow):
    for alg in controlFlow[seqName]:
        #print(alg)
        algName = alg.split( '/' )[ 1 ] if '/' in alg else alg
        algType = alg.split( '/' )[ 0 ] if '/' in alg else 'Algorithm'

        if algType == 'AthSequencer':
            fillSequence(seq, alg, controlFlow)
        elif not algName in seq:
            seq.append(algName)

def fillConsumes(algos, dataFlow):
    consumes = {}
    not_on_any_input = set(algos)
    outputs = {}
    for algName in algos:
        for inNode, outNode in dataFlow.out_edges(algName):
            if outNode in outputs:
                raise Exception("Output node %s was already in the outputs" % outNode)
            outputs[outNode] = algName
    for algName in algos:
        s = set()
        for inNode, outNode in dataFlow.in_edges(algName):
            inAlg = outputs[inNode]
            not_on_any_input.discard(inAlg)
            s.add(inAlg)
        consumes[sanitize(algName)] = sanitize(s)
    consumes["_out"] = sanitize(not_on_any_input)
    return consumes

def fillDeclarations(algos):
    decls = {}
    for algName in algos:
        decls[sanitize(algName)] = "SimCPU"
    return decls

def fillDefinitions(algos, times):
    defs = {}
    for algName in algos:
        try:
            timeInS = times[algName]
        except KeyError:
            timeInS = 0
            print("No time information for module %s" % algName)
        timeInNs = int(timeInS*1e9)
        defs[sanitize(algName)] = dict(
            produce = [dict(
                name = "cpu",
                values = [timeInNs],
                unit = "ns"
            )]
        )
    return defs

def main(opts):
    config = {}

    controlFlow = nx.read_graphml(opts.cflow)
    dataFlow = nx.read_graphml(opts.dflow)
    seq = []
    fillSequence(seq, 'AthSequencer/AthAlgSeq', controlFlow)
    config["moduleSequence"] = sanitize(seq)

    config["moduleConsumes"] = fillConsumes(seq, dataFlow)

    times = {}
    with open(opts.time) as f:
        times = json.load(f)
    
    config["moduleDeclarations"] = fillDeclarations(seq)
    config["moduleDefinitions"] = fillDefinitions(seq, times)

    with open(opts.output, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert files from ATLAS simulation")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output file")
    parser.add_argument("-t", "--time", type=str, required=True,
                        help="Timing input file")
    parser.add_argument("-c", "--cflow", type=str, required=True,
                        help="Control flow graph")
    parser.add_argument("-d", "--dflow", type=str, required=True,
                        help="Data flow graph")

    opts = parser.parse_args()

    main(opts)


