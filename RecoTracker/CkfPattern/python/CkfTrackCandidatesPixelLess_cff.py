import FWCore.ParameterSet.Config as cms

# TrackerTrajectoryBuilders
from RecoTracker.CkfPattern.CkfTrajectoryBuilder_cff import *
import RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi
# TrajectoryCleaning
from TrackingTools.TrajectoryCleaning.TrajectoryCleanerBySharedHits_cfi import *
# navigation school
from RecoTracker.TkNavigation.NavigationSchoolESProducer_cff import *

# generate CTF track candidates ############
import RecoTracker.CkfPattern.CkfTrackCandidates_cfi 
ckfTrackCandidatesPixelLess = RecoTracker.CkfPattern.CkfTrackCandidates_cfi.ckfTrackCandidates.clone(
    TrajectoryBuilder = RecoTracker.CkfPattern.GroupedCkfTrajectoryBuilder_cfi.GroupedCkfTrajectoryBuilder,
    src = cms.InputTag('globalPixelLessSeeds')
    )



