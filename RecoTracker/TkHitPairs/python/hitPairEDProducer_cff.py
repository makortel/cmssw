from Configuration.StandardSequences.Eras import eras
from RecoTracker.TkHitPairs.hitPairEDProducer_cfi import *
eras.trackingPhase1PU70.toModify(hitPairEDProducer, maxElement=0)
eras.trackingPhase2PU140.toModify(hitPairEDProducer, maxElement=0)
