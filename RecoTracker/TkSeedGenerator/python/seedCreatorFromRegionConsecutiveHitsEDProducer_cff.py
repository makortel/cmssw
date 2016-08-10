from Configuration.StandardSequences.Eras import eras
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsEDProducer_cfi import *
eras.trackingPhase1PU70.toModify(seedCreatorFromRegionConsecutiveHitsEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
eras.trackingPhase2PU140.toModify(seedCreatorFromRegionConsecutiveHitsEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
