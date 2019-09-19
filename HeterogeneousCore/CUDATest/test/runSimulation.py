#!/usr/bin/env python3

import re
import sys
import json
import argparse
import subprocess

nev_quantum = 100
nev_per_stream = 1*nev_quantum

times = 1
n_streams_threads = [(0, i) for i in range(1,2)]
#n_streams_threads = [(4*i, i) for i in xrange(1,9)]
events_re = re.compile("TrigReport Events total = (?P<events>\d+) passed")
time_re = re.compile("event loop Real/event = (?P<time>\d+.\d+)")
#time_re = re.compile("event loop Real/event")

def seconds(m):
    return ( float(m.group("hour"))*60 + float(m.group("min")) )*60 + float(m.group("sec"))

def throughput(output):
    nevents = None
    time = None
    for line in output.splitlines():
        if nevents is None:
            m = events_re.search(line)
            if m:
                nevents = int(m.group("events"))
        else:
            m = time_re.search(line)
            if m:
                time = float(m.group("time"))

    if nevents is None:
        raise Exception("Did not find number of events")
    if time is None:
        raise Exception("Did not find time/event")
    thr = 1./time
    
    print("Processed %d events in %f seconds, throuhgput %s ev/s" % (nevents, time*nevents, thr))
    return thr

def run(nev, nstr, nth, config):
    cmssw = subprocess.Popen(["cmsRun", config, "maxEvents=%d"%nev, "numberOfStreams=%d"%nstr, "numberOfThreads=%d"%nth], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    (stdout, stderr) = cmssw.communicate()
    if cmssw.returncode != 0:
        raise Exception("Got return code %d, output\n%s" % (cmssw.returncode, stdout))
    return throughput(stdout)

def main(opts):
    results = []

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        nev = nev_per_stream*nstr
        print("Number of streams %d threads %d events %d" % (nstr, nth, nev))
        thrs = []
        for i in range(times):
            thrs.append(run(nev, nstr, nth, opts.config))
        print("Number of streams %d threads %d, average throughput %f" % (nstr, nth, (sum(thrs)/len(thrs))))
        results.append(dict(
            threads=nth,
            straems=nstr,
            events=nev,
            throughput=sum(thrs)/len(thrs)
        ))
    data = dict(
        config=opts.config,
        results=results
    )
    with open(opts.output, "w") as out:
        json.dump(data, out, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate vector addition loop timing for CUDA simulation")
    parser.add_argument("config", type=str,
                        help="CMSSW configuration file to run")
    parser.add_argument("-o", "--output", type=str, default="result.json",
                        help="Output JSON file")
    opts = parser.parse_args()
    main(opts)
