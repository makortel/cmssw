#ifndef RecoTracker_SeedingLayerSet_SeedingLayerSetNew
#define RecoTracker_SeedingLayerSet_SeedingLayerSetNew

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>

class SeedingLayerSetNew {
public:
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;

  SeedingLayerSetNew();
  explicit SeedingLayerSetNew(unsigned int nlayers_);
  ~SeedingLayerSetNew();

  void addLayersHits(const std::vector<std::vector<ConstRecHitPointer> >& hits);

  unsigned int nlayers() const { return nlayers_; }
  unsigned int nlayerSets() const { return layerIndices_.size() / nlayers_; }

  std::vector<ConstRecHitPointer> hits(unsigned int layerSetIndex, unsigned int layerIndex) const;

private:
  unsigned int nlayers_;
  std::vector<unsigned int> layerIndices_;
  std::vector<ConstRecHitPointer> rechits_;
};

#endif
