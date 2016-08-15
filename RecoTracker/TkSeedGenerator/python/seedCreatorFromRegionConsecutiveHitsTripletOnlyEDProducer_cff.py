from Configuration.StandardSequences.Eras import eras
from RecoTracker.TkSeedGenerator.seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer_cfi import *
eras.trackingPhase1PU70.toModify(seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
eras.trackingPhase2PU140.toModify(seedCreatorFromRegionConsecutiveHitsTripletOnlyEDProducer,
   magneticField = '',
   propagator = 'PropagatorWithMaterial',
)
