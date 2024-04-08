import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

import argparse
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Test ModuleTypeResolver and Ref')
parser.add_argument("--input", action="append", default=[], help="Input files. Use .root suffix for ROOT files, and .dat suffix for streamer files")
args = parser.parse_args()
if len(args.input) == 0:
    parser.error("No input files")

files = ["file:"+f for f in args.input]
if args.input[0][-4:] == ".dat":
    process.source = cms.Source("NewEventStreamFileReader",
        fileNames = cms.untracked.vstring(files)
    )
else:
    process.source = cms.Source("PoolSource",
        fileNames = cms.untracked.vstring(files)
    )

process.tester = cms.EDAnalyzer("OtherThingAnalyzer",
    other = cms.untracked.InputTag("otherThing","testUserTag")
)

process.e = cms.EndPath(process.tester)
