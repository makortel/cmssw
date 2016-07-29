#ifndef RecoTracker_TkHitPairs_RegionsSeedingHitSets_H
#define RecoTracker_TkHitPairs_RegionsSeedingHitSets_H

#include "RecoTracker/TkSeedingLayers/interface/SeedingHitSet.h"
#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"

// defined in this package instead of RecoTracker/TkSeedingLayers to avoid circular dependencies

class RegionsSeedingHitSets {
public:
  using RegionIndex = ihd::RegionIndex;

  RegionsSeedingHitSets() = default;
  ~RegionsSeedingHitSets() = default;

  void swap(RegionsSeedingHitSets& rh) {
    regions_.swap(rh.regions_);
    hitSets_.swap(rh.hitSets_);
  }

  void reserve(size_t nregions, size_t nhitsets) {
    regions_.reserve(nregions);
    hitSets_.reserve(nhitsets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    hitSets_.shrink_to_fit();
  }

  void beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, hitSets_.size());
  }

  template <typename... Args>
  void emplace_back(Args&&... args) {
    hitSets_.emplace_back(std::forward<Args>(args)...);
    regions_.back().setLayerSetsEnd(hitSets_.size());
  }

  size_t regionSize() const { return regions_.size(); }
  size_t size() const { return hitSets_.size(); }

private:
  std::vector<RegionIndex> regions_;
  std::vector<SeedingHitSet> hitSets_;
};

#endif
