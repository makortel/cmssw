#!/usr/bin/env python3

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import json

def makePlot(inputfiles, output):
    fig, axs = plt.subplots(1, 1)
    axs = [axs]
    axs[0].set(ylabel = "Throughput (events/s)")
    #axs[1].set(ylabel = "Speedup wrt. baseline 1 thread")
    axs[0].set(xlabel = "Threads/EDM streams")
    axs[0].grid()
    #axs[1].grid()

    baseline = None

    for fname, label in inputfiles:
        data = None
        with open(fname) as f:
            data = json.load(f)
        threads = [x["threads"] for x in data["results"]]
        streams = [x["streams"] for x in data["results"]]
        throughput = [x["throughput"] for x in data["results"]]
        if baseline is None:
            baseline = throughput[0]
        speedup = [x / baseline for x in throughput]

        if streams != threads:
            raise Exception("streams are not equal to threads")
        axs[0].plot(threads, throughput, ".-", label=label)
        #axs[1].plot(threads, speedup, ".-", label=label)

    axs[0].set_ylim(bottom=0)
    axs[0].legend()

    fig.savefig(output)
    print(output)

def main():
    makePlot([
        ("baseline.json", "Baseline"),
    ], "baseline.png")
    makePlot([
        ("baseline.json", "Baseline"),
        ("all_externalwork.json", "All GPU modules ExternalWork")
    ], "all_externalwork.png")

if __name__ == "__main__":
    main()
