#!/usr/bin/env python3

import os
import re
import sys
import json
import time
import signal
import argparse
import subprocess
import multiprocessing

# felk40
cores_felk40 = [1, 2, 3, 0]

# online
# core 0 as the last as it usually has the OS
cores_online = list(range(16,32)) + list(range(1,16)) + [0]
cores_online_ht = cores_online + list(range(48,64)) + list(range(32,48))


background_time = 4*60*60
# felk40: 1700 ev/s on 8 threads, 
nev_quantum = 4000
#nev_per_stream = 300*nev_quantum
nblocks_per_stream = {
    1: 68,
    2: 60,
    3: 6,
    4: 4,
    5: 2,
    6: 5,
    7: 5,
}

times = 1
events_re = re.compile("TrigReport Events total = (?P<events>\d+) passed")
time_re = re.compile("event loop Real/event = (?P<time>\d+.\d+)")

def printMessage(*args):
    print(time.strftime("%y-%m-%d %H:%M:%S"), *args)

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
    
    printMessage("Processed %d events in %f seconds, throuhgput %s ev/s" % (nevents, time*nevents, thr))
    return (thr, time*nevents)

def partition_cores(cores, nth):
    if nth >= len(cores):
        return (cores, [])

    return (cores[0:nth], cores[nth:])

def run(nev, nstr, cores_main, opts, logfilename):
    nth = len(cores_main)
    with open(logfilename, "w") as logfile:
        taskset = []
        nvprof = []
        cmsRun = ["cmsRun", opts.config, "maxEvents=%d"%nev, "numberOfStreams=%d"%nstr, "numberOfThreads=%d"%nth] + opts.args
        if opts.histoFileName is not None:
            with open(opts.histoFileName, "a+") as histofile:
                histofile.write("\nnstr%d_nth%d\n" % (nstr, nth))
            cmsRun.append("histoFileName="+opts.histoFileName)
        if opts.taskset:
            taskset = ["taskset", "-c", ",".join(cores_main)]
        if opts.nvprof:
            nvprof = ["nvprof", "-o", logfilename.replace("_log_", "_prof_").replace(".txt", ".nvvp")]

        logfile.write(" ".join(taskset+nvprof+cmsRun))
        logfile.write("\n----\n")
        logfile.flush()
        cmssw = subprocess.Popen(taskset+nvprof+cmsRun, stdout=logfile, stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            cmssw.wait()
        except KeyboardInterrupt:
            try:
                cmssw.terminate()
            except OSError:
                pass
            cmssw.wait()
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
    #cores = cores_felk40
    cores = cores_online # force one thread per physical core, ignore HT
    if opts.useHT:
        #cores = list(range(0, multiprocessing.cpu_count()))
        cores = cores_online_ht
    cores = [str(x) for x in cores]
    #cores = cores[:48]

    maxThreads = len(cores)
    if opts.maxThreads > 0:
        maxThreads = min(maxThreads, opts.maxThreads)

    nthreads = range(opts.minThreads,maxThreads+1)
    if len(opts.numThreads) > 0:
        nthreads = [x for x in opts.numThreads if x >= opts.minThreads and x <= maxThreads]
    n_streams_threads = [(i*opts.streamsPerThread, i) for i in nthreads]
    if len(opts.numStreams) > 0:
        n_streams_threads = [(s, t) for t in nthreads for s in opts.numStreams]

    nev_per_stream = opts.eventsPerStream
    if nev_per_stream is None:
        eventBlocksPerStream = nblocks_per_stream.get(opts.variant, None)
        if eventBlocksPerStream is None:
            raise Exception("No default number of event blocks for variant %d, and --eventsPerStream was not given" % opts.variant)
        nev_per_stream = eventBlocksPerStream * nev_quantum

    data = dict(
        config=opts.config,
        args=" ".join(opts.args),
        results=[]
    )
    outputJson = opts.output+".json"

    alreadyExists = set()
    if not opts.overwrite and os.path.exists(outputJson):
        with open(outputJson) as inp:
            data = json.load(inp)
    if not opts.append:
        for res in data["results"]:
            alreadyExists.add( (res["streams"], res["threads"]) )

    stop = False
    keep_nev = None

    for nstr, nth in n_streams_threads:
        if nstr == 0:
            nstr = nth
        if (nstr, nth) in alreadyExists:
            continue

        nev = nev_per_stream*nstr
        if keep_nev is not None:
            nev = keep_nev
        (cores_main, cores_bkg) = partition_cores(cores, nth)

        thrs = []
        with open(opts.output+"_log_bkg_nstr%d_nth%d.txt"%(nstr, nth), "w") as bkglogfile:
            cmsswBackground = launchBackground(opts, cores_bkg, bkglogfile)
            if cmsswBackground is not None:
                msg = "Background CMSSW pid %d" % cmsswBackground.pid
                if opts.taskset:
                    msg += ", running on cores %s" % ",".join(cores_bkg)
                printMessage(msg)

            try:
                msg = "Number of streams %d threads %d events %d" % (nstr, nth, nev)
                if opts.taskset:
                    msg += ", running on cores %s" % ",".join(cores_main)
                printMessage(msg)
                for i in range(times):
                    (th, wtime) = run(nev, nstr, cores_main, opts, opts.output+"_log_nstr%d_nth%d_n%d.txt"%(nstr, nth, i))
                    thrs.append(th)
                    if opts.stopIncreasingEventBlocksAfterWallTime > 0 and wtime > opts.stopIncreasingEventBlocksAfterWallTime:
                        keep_nev = nev
                    if opts.stopAfterWallTime > 0 and wtime > opts.stopAfterWallTime:
                        stop = True
                        break
            finally:
                if cmsswBackground is not None:
                    printMessage("Run complete, terminating background CMSSW, waiting for 10 seconds")
                    cmsswBackground.send_signal(signal.SIGUSR2)
                    try:
                        cmsswBackground.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        try:
                            printMessage("Terminating background CMSSW nicely timed out, terminating in the hard way")
                            cmsswBackground.terminate()
                        except OSError:
                            pass
                        cmsswBackground.wait()

        printMessage("Number of streams %d threads %d, average throughput %f" % (nstr, nth, (sum(thrs)/len(thrs))))
        print()
        data["results"].append(dict(
            threads=nth,
            streams=nstr,
            events=nev,
            throughput=sum(thrs)/len(thrs)
        ))
        # Save results after each test
        with open(outputJson, "w") as out:
            json.dump(data, out, indent=2)
        if stop:
            print("Reached max wall time of %d s, stopping scan" % opts.stopAfterWallTime)
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate vector addition loop timing for CUDA simulation")
    parser.add_argument("config", type=str,
                        help="CMSSW configuration file to run")
    parser.add_argument("-o", "--output", type=str, default="result",
                        help="Prefix of output JSON and log files. If the output JSON file exists, it will be updated (see also --overwrite) (default: 'result')")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite the output JSON instead of updating it")
    parser.add_argument("--append", action="store_true",
                        help="Append new (stream, threads) results insteads of ignoring already existing point")
    parser.add_argument("--nvprof", action="store_true",
                        help="Run the main program through nvprof")
    parser.add_argument("--taskset", action="store_true",
                        help="Use taskset to explicitly set the cores where to run on")
    parser.add_argument("--useHT", action="store_true",
                        help="Use all logical cores instead. Default is to use physical cores only, i.e. ignore the HT logical core")
    parser.add_argument("--no-background", dest="background", action="store_false",
                        help="Disable background process occupying the other cores")
    parser.add_argument("--minThreads", type=int, default=1,
                        help="Minimum number of threads to use in the scan (default: 1)")
    parser.add_argument("--maxThreads", type=int, default=-1,
                        help="Maximum number of threads to use in the scan (default: -1 for the number of cores)")
    parser.add_argument("--numThreads", type=str, default="",
                        help="Comma separated list of numbers of threads to use in the scan (default: empty for all)")
    parser.add_argument("--streamsPerThread", type=int, default=1,
                        help="Number of streams per thread (default 1)")
    parser.add_argument("--numStreams", type=str, default="",
                        help="Comma separated list of numbers of streams to use in the scan (default: empty for always the same as the number of threads). If both number of threads and number of streams have more than 1 element, a 2D scan is done with all the combinations")
    parser.add_argument("--variant", type=int, default=1,
                        help="Application variant, can be one of 1,2,3,4,5 (default: 1)")
    parser.add_argument("--eventsPerStream", type=int, default=None,
                        help="Number of events to be used per EDM stream (default: 85*4kev for variant 1)")
    parser.add_argument("--stopIncreasingEventBlocksAfterWallTime", type=int, default=-1,
                        help="Stop increasing number of event blocks after the wall time of the job reaches this many seconds (default: -1 for no limit)")
    parser.add_argument("--stopAfterWallTime", type=int, default=-1,
                        help="Stop running after the wall time of the job reaches this many in seconds (default: -1 for no limit)")
    parser.add_argument("--histoFileName", type=str, default=None,
                        help="Pass the string to histoFileName argument with numbers of streams and threads appended")

    parser.add_argument("args", nargs=argparse.REMAINDER)

    opts = parser.parse_args()
    if opts.minThreads <= 0:
        parser.error("minThreads must be > 0, got %d" % opts.minThreads)
    if opts.maxThreads <= 0 and opts.maxThreads != -1:
        parser.error("maxThreads must be > 0 or -1, got %d" % opts.maxThreads)
    if opts.numThreads != "":
        opts.numThreads = [int(x) for x in opts.numThreads.split(",")]
    if opts.numStreams != "":
        opts.numStreams = [int(x) for x in opts.numStreams.split(",")]
    if opts.variant not in [0,1,2,3,4,5,6,7,99]:
        parser.error("Invalid variant %d" % opts.variant)
    opts.args.append("variant=%d"%opts.variant)

    main(opts)
