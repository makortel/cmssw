#!/usr/bin/env python

import re
import sys
import subprocess

times = 10
n_streams_threads = [(0, i) for i in xrange(1,9)]
#n_streams_threads = [(4*i, i) for i in xrange(1,9)]
times = 1
nthreads = xrange(1,2)
stamp_re = re.compile("Begin processing the .* Event (?P<event>\d+),.*(?P<hour>\d\d):(?P<min>\d\d):(?P<sec>\d\d\.\d\d\d)")

def seconds(m):
    return ( float(m.group("hour"))*60 + float(m.group("min")) )*60 + float(m.group("sec"))

def throughput(output):
    eventTimes = []
    for line in output.split("\n"):
        m = stamp_re.search(line);
        if m:
            eventTimes.append( (int(m.group("event")), seconds(m)) )

    if len(eventTimes) < 3:
        raise Exception("Did not find times from output\n"+output)
    
    nevents = eventTimes[-1][0] - eventTimes[1][0]
    time = eventTimes[-1][1] - eventTimes[1][1]
    thr = nevents/time

    print "Processed %d events in %f seconds, throuhgput %s ev/s" % (nevents, time, thr)
    return thr

def run(nstr, nth, config):
    cmssw = subprocess.Popen(["cmsRun", config, "numberOfStreams=%d"%nstr, "numberOfThreads=%d"%nth], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout, stderr) = cmssw.communicate()
    if cmssw.returncode != 0:
        raise Exception("Got return code %d, output\n%s" % (cmssw.returncode, stdout))
    return throughput(stdout)

def main(config):
    for nstr, nth in n_streams_threads:
        print "Number of streams %d threads %d" % (nstr, nth)
        thrs = []
        for i in xrange(times):
            thrs.append(run(nstr, nth, config))
        print "Number of streams %d threads %d, average throughput %f" % (nstr, nth, (sum(thrs)/len(thrs)))

if __name__ == "__main__":
    main(sys.argv[1])
