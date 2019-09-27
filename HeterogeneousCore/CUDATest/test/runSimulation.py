#!/usr/bin/env python3

import re
import sys
import json
import signal
import argparse
import subprocess
import multiprocessing

# felk40
cores_felk40 = [0, 1, 2, 3]

background_time = 60*60
# felk40: 1700 ev/s on 8 threads, 
nev_quantum = 4000
#nev_per_stream = 300*nev_quantum
nev_per_stream = 85*nev_quantum

times = 1
events_re = re.compile("TrigReport Events total = (?P<events>\d+) passed")
time_re = re.compile("event loop Real/event = (?P<time>\d+.\d+)")

def seconds(m):
    return ( float(m.group("hour"))*60 + float(m.group("min")) )*60 + float(m.group("sec"))

def throughput(output):
    nevents = None
    time = None
    for line in output:
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

def partition_cores(cores, nth):
    if nth >= len(cores):
        return (cores, [])

    return (cores[1:nth+1], [cores[0]] + cores[nth+1:])

def run(nev, nstr, cores_main, opts, logfilename):
    nth = len(cores_main)
    with open(logfilename, "w") as logfile:
        taskset = []
        nvprof = []
        cmsRun = ["cmsRun", opts.config, "maxEvents=%d"%nev, "numberOfStreams=%d"%nstr, "numberOfThreads=%d"%nth]
        if opts.taskset:
            taskset = ["taskset", "-c", ",".join(cores_main)]
        if opts.nvprof:
            nvprof = ["nvprof", "-o", logfilename.replace("_log_", "_prof_").replace(".txt", ".nvvp")]

        cmssw = subprocess.Popen(taskset+nvprof+cmsRun, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True)
        cmssw.communicate()
        if cmssw.returncode != 0:
            raise Exception("Got return code %d, see output in the log file %s" % (cmssw.returncode, logfilename))
    with open(logfilename) as logfile:
        return throughput(logfile)

def launchBackground(opts, cores_bkg, logfile):
    if not opts.background:
        return None
    nth = len(cores_bkg)
    if nth == 0:
        return None
    evs = background_time * nth
    taskset = []
    cmsRun = ["cmsRun", "HeterogeneousCore/CUDATest/test/cpucruncher_cfg.py", "maxEvents=%d"%evs, "numberOfStreams=%d"%nth, "numberOfThreads=%d"%nth]
    if opts.taskset:
        taskset = ["taskset", "-c", ",".join(cores_bkg)]

    cmssw = subprocess.Popen(taskset+cmsRun, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True)
    return cmssw

def main(opts):
    results = []

    cores = list(range(0, multiprocessing.cpu_count()))
    if opts.taskset:
        cores = cores_felk40
    cores = [str(x) for x in cores]

    maxThreads = len(cores)
    if opts.maxThreads > 0:
        maxThreads = min(maxThreads, opts.maxThreads)

    nthreads = range(opts.minThreads,maxThreads+1)
    if len(opts.numThreads) > 0:
        nthreads = [x in opts.numThreads if x >= opts.minThreads and x <= opts.maxThreads]
    n_streams_threads = [(i, i) for i in nthreads]

    data = dict(
        config=opts.config,
        results=[]
    )

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        nev = nev_per_stream*nstr
        (cores_main, cores_bkg) = partition_cores(cores, nth)

        thrs = []
        with open(opts.output+"_log_bkg_nstr%d_nth%d.txt"%(nstr, nth), "w") as bkglogfile:
            cmsswBackground = launchBackground(opts, cores_bkg, bkglogfile)
            if cmsswBackground is not None:
                msg = "Background CMSSW pid %d" % cmsswBackground.pid
                if opts.taskset:
                    msg += ", running on cores %s" % ",".join(cores_bkg)
                print(msg)

            try:
                msg = "Number of streams %d threads %d events %d" % (nstr, nth, nev)
                if opts.taskset:
                    msg += ", running on cores %s" % ",".join(cores_main)
                print(msg)
                for i in range(times):
                    thrs.append(run(nev, nstr, cores_main, opts, opts.output+"_log_nstr%d_nth%d_n%d.txt"%(nstr, nth, i)))
            finally:
                if cmsswBackground is not None:
                    print("Run complete, terminating background CMSSW")
                    cmsswBackground.send_signal(signal.SIGUSR2)

        print("Number of streams %d threads %d, average throughput %f" % (nstr, nth, (sum(thrs)/len(thrs))))
        print()
        results.append(dict(
            threads=nth,
            streams=nstr,
            events=nev,
            throughput=sum(thrs)/len(thrs)
        ))
        # Save results after each test
        data["results"] = results
        with open(opts.output+".json", "w") as out:
            json.dump(data, out, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate vector addition loop timing for CUDA simulation")
    parser.add_argument("config", type=str,
                        help="CMSSW configuration file to run")
    parser.add_argument("-o", "--output", type=str, default="result",
                        help="Prefix of output JSON and log files (default: 'result')")
    parser.add_argument("--nvprof", action="store_true",
                        help="Run the main program through nvprof")
    parser.add_argument("--taskset", action="store_true",
                        help="USe taskset to explicitly set the cores where to run on")
    parser.add_argument("--no-background", dest="background", action="store_false",
                        help="Disable background process occupying the other cores")
    parser.add_argument("--minThreads", type=int, default=1,
                        help="Minimum number of threads to use in the scan (default: 1)")
    parser.add_argument("--maxThreads", type=int, default=-1,
                        help="Maximum number of threads to use in the scan (default: -1 for the number of cores)")
    parser.add_argument("--numThreads", type=str, default="",
                        help="Comma separated list of numbers threads to use in the scan (default: empty for all)")
    opts = parser.parse_args()
    if opts.minThreads <= 0:
        parser.error("minThreads must be > 0, got %d" % opts.minThreads)
    if opts.maxThreads <= 0 and opts.maxThreads != -1:
        parser.error("maxThreads must be > 0 or -1, got %d" % opts.maxThreads)
    if opts.numThreads != "":
        opts.numThreads = [int(x) for x in opts.numThreads.split(",")]

    main(opts)
