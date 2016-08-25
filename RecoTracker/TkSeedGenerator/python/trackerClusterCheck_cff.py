from Configuration.StandardSequences.Eras import eras
from RecoTracker.TkSeedGenerator.trackerClusterCheck_cfi import *
# Disable too many clusters check until we have an updated cut string for phase1 and phase2
eras.phase1Pixel.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME
eras.phase2_tracker.toModify(trackerClusterCheck, doClusterCheck=False) # FIXME
