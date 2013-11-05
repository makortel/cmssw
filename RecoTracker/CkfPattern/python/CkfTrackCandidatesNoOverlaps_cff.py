import FWCore.ParameterSet.Config as cms

# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cff import *
import RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
# navigation school
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *
import copy
from RecoTracker.CkfPattern.CkfTrackCandidates_cfi import *
# generate CTF track candidates ############
ckfTrackCandidatesNoOverlaps = copy.deepcopy(ckfTrackCandidates)
ckfTrackCandidatesNoOverlaps.TrajectoryBuilder = RecoTracker.CkfPattern.CkfTrajectoryBuilder_cfi.CkfTrajectoryBuilder

