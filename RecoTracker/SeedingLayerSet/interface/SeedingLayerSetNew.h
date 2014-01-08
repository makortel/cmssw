#ifndef RecoTracker_SeedingLayerSet_SeedingLayerSetNew
#define RecoTracker_SeedingLayerSet_SeedingLayerSetNew

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>

class SeedingLayerSetNew {
public:
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  SeedingLayerSetNew();
  ~SeedingLayerSetNew();

private:
  std::vector<ConstRecHitPointer> rechits_;
};

#endif
