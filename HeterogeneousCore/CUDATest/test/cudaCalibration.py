#!/usr/bin/env python3

import os
import sys
import json
import sqlite3
import argparse
import subprocess
import statistics

def main(opts):
#    iters = [0, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048,
#             2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
#             9216, 10240, 12288, 14336, 16384, 20480, 28672, 32768,
#             49152, 65536, 98304, 131072,
# extrapolation seems to work fine
#             262144,
#             524288,
#             1048576,
#             2097152,
#             4194304,
#             8388608,
#             16777216,
#             33554432,
#             67108864,
#             134217728,
#             268435456,
#             536 870 912
#    ]
    iters = [0, 5, 10, 20, 40, 60, 80, 100,
             150, 200, 250, 300, 350, 400, 450, 500,
             600, 700, 800, 900,
             1000, 1500, 2000, 2500, 3000, 4000, 5000,  6000, 8000,
             10000, 15000, 20000, 30000, 40000, 50000, 75000,
             100000, 150000, 200000, 300000, 400000, 500000, 750000,
             1000000, 2000000, 4000000, 8000000,
             15000000, 30000000, 60000000,
             100000000, 200000000, 400000000, 800000000,
             1600000000,
    ]


    p = subprocess.Popen(["nvprof", "-f", "-o", "cudaCalibration.nvvp", "cudaCalibration"] + [str(x) for x in iters], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (stdout, stderr) = p.communicate()
    if p.returncode != 0:
        print("nvprof cudaCalibration command failed with exit code %d" % p.returncode)
        print(stdout)
        return

    conn = sqlite3.connect("cudaCalibration.nvvp")
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r[0]] = r[1]

    times = []
    for r in conn.execute("SELECT CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.start, CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.end, name FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL"):
        times.append(r[1]-r[0])

    if len(times) != 4*len(iters) + 4:
        raise Exception("Unexpected number of kernel calls %d, expected %d" % (len(times), 4*len(iters) + 4))

    # ignore first 4, they're warmup
    times = times[4:]

    # the remaining are averages of 4
    times_avg = []
    for i, n in enumerate(iters):
        t = [times[i], times[i+len(iters)], times[i+2*len(iters)], times[i+3*len(iters)]]
        t = [x*1e-3 for x in t]
        avg = statistics.mean(t)
        times_avg.append(avg)
        err = statistics.stdev(t)/len(t)
        print("Iters %d mean %f +- %f us (%.4e)" % (n, avg, err, err/avg))

    data = {"niters": iters,
            "timesInMicroSeconds": times_avg}
    with open("cudaCalibration.json", "w") as out:
        json.dump(data, out, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate vector addition loop timing for CUDA simulation")
    opts = parser.parse_args()
    main(opts)

