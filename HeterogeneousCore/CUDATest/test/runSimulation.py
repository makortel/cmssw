#!/usr/bin/env python3

import re
import sys
import json
import argparse
import subprocess

# felk40
cores = [0, 1, 2, 3]
cores = [str(x) for x in cores]
background_time = 10*60

nev_quantum = 100
nev_per_stream = 100*nev_quantum

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

def partition_cores(nth):
    if nth >= len(cores):
        return (cores, [])

    return (cores[1:nth+1], [cores[0]] + cores[nth+1:])

def run(nev, nstr, cores_main, config):
    nth = len(cores_main)
    cmssw = subprocess.Popen(["taskset", "-c", ",".join(cores_main), "cmsRun", config, "maxEvents=%d"%nev, "numberOfStreams=%d"%nstr, "numberOfThreads=%d"%nth], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    (stdout, stderr) = cmssw.communicate()
    if cmssw.returncode != 0:
        raise Exception("Got return code %d, output\n%s" % (cmssw.returncode, stdout))
    return throughput(stdout)

def launchBackground(cores_bkg):
    nth = len(cores_bkg)
    evs = background_time * nth
    cmssw = subprocess.Popen(["taskset", "-c", ",".join(cores_bkg), "cmsRun", "HeterogeneousCore/CUDATest/test/cpucruncher_cfg.py", "maxEvents=%d"%evs, "numberOfStreams=%d"%nth, "numberOfThreads=%d"%nth], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return cmssw

def main(opts):
    results = []

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        nev = nev_per_stream*nstr
        (cores_main, cores_bkg) = partition_cores(nth)

        cmsswBackground = launchBackground(cores_bkg)
        print("Background CMSSW pid %d, running on cores %s" % (cmsswBackground.pid, ",".join(cores_bkg)))

        print("Number of streams %d threads %d events %d, running on cores %s" % (nstr, nth, nev, ",".join(cores_main)))
        thrs = []
        for i in range(times):
            thrs.append(run(nev, nstr, cores_main, opts.config))
        cmsswBackground.kill()

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
