#ifndef RecoTracker_TkHitPairs_IntermediateHitDoublets_h
#define RecoTracker_TkHitPairs_IntermediateHitDoublets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

/**
 * Simple container of temporary information delivered from hit pair
 * generator to hit triplet generator via edm::Event.
 */
class IntermediateHitDoublets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;

  IntermediateHitDoublets(): seedingLayers_(nullptr) {}
  explicit IntermediateHitDoublets(const SeedingLayerSetsHits *seedingLayers): seedingLayers_(seedingLayers) {}
  IntermediateHitDoublets(const IntermediateHitDoublets& rh); // only to make ROOT dictionary generation happy
  ~IntermediateHitDoublets() = default;

  void swap(IntermediateHitDoublets& rh) {
    std::swap(regions_, rh.regions_);
    std::swap(layerPairs_, rh.layerPairs_);
  }

  void reserve(size_t nregions, size_t nlayersets) {
    regions_.reserve(nregions);
    layerPairs_.reserve(nregions*nlayersets);
  }

  void shrink_to_fit() {
    regions_.shrink_to_fit();
    layerPairs_.shrink_to_fit();
  }

  void beginRegion(const TrackingRegion *region) {
    regions_.emplace_back(region, layerPairs_.size());
  }

  void addDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets, LayerHitMapCache&& cache) {
    layerPairs_.emplace_back(layerSet, std::move(doublets), std::move(cache));
  }

private:
  const SeedingLayerSetsHits *seedingLayers_;

  struct RegionIndex {
    RegionIndex(const TrackingRegion *reg, unsigned int ind): region_(reg), layerPairIndex_(ind) {}

    const TrackingRegion *region_;
    unsigned int layerPairIndex_;  /// index to doublets_, pointing to the beginning of the layer pairs of this region
  };

  struct LayerPairHitDoublets {
    LayerPairHitDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets, LayerHitMapCache&& cache):
      layerPair_(layerSet[0].index(), layerSet[1].index()),
      doublets_(std::move(doublets)),
      cache_(std::move(cache))
    {}

    LayerPair layerPair_;
    HitDoublets doublets_;
    LayerHitMapCache cache_;
  };

  std::vector<RegionIndex> regions_;
  std::vector<LayerPairHitDoublets> layerPairs_;
};

#endif
