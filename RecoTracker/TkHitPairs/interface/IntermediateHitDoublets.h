#ifndef RecoTracker_TkHitPairs_IntermediateHitDoublets_h
#define RecoTracker_TkHitPairs_IntermediateHitDoublets_h

#include "RecoTracker/TkHitPairs/interface/LayerHitMapCache.h"
#include "TrackingTools/TransientTrackingRecHit/interface/SeedingLayerSetsHits.h"

namespace ihd {
  class RegionIndex {
  public:
    RegionIndex(const TrackingRegion *reg, unsigned int ind): region_(reg), layerSetIndex_(ind) {}

    const TrackingRegion& region() const { return *region_; }
    unsigned int layerSetIndex() const { return layerSetIndex_; }

  private:
    const TrackingRegion *region_;
    unsigned int layerSetIndex_;  /// index to doublets_, pointing to the beginning of the layer pairs of this region
  };
}

/**
 * Simple container of temporary information delivered from hit pair
 * generator to hit triplet generator via edm::Event.
 */
class IntermediateHitDoublets {
public:
  using LayerPair = std::tuple<SeedingLayerSetsHits::LayerIndex, SeedingLayerSetsHits::LayerIndex>;
  using RegionIndex = ihd::RegionIndex;

  class LayerPairHitDoublets {
  public:
    LayerPairHitDoublets(const SeedingLayerSetsHits::SeedingLayerSet& layerSet, HitDoublets&& doublets, LayerHitMapCache&& cache):
      layerPair_(layerSet[0].index(), layerSet[1].index()),
      doublets_(std::move(doublets)),
      cache_(std::move(cache))
    {}

    SeedingLayerSetsHits::LayerIndex innerLayerIndex() const { return std::get<0>(layerPair_); }
    SeedingLayerSetsHits::LayerIndex outerLayerIndex() const { return std::get<1>(layerPair_); }

    const HitDoublets& doublets() const { return doublets_; }
    const LayerHitMapCache& cache() const { return cache_; }

  private:
    LayerPair layerPair_;
    HitDoublets doublets_;
    LayerHitMapCache cache_;
  };

  ////////////////////

  class RegionLayerHits {
  public:
    using const_iterator = std::vector<LayerPairHitDoublets>::const_iterator;

    RegionLayerHits(const TrackingRegion* region, const_iterator begin, const_iterator end):
      region_(region), layerPairsBegin_(begin), layerPairsEnd_(end) {}

    const TrackingRegion& region() const { return *region_; }

    const_iterator begin() const { return layerPairsBegin_; }
    const_iterator cbegin() const { return begin(); }
    const_iterator end() const { return layerPairsEnd_; }
    const_iterator cend() const { return end(); }

  private:
    const TrackingRegion *region_;
    const const_iterator layerPairsBegin_;
    const const_iterator layerPairsEnd_;
  };

  ////////////////////

  class const_iterator {
  public:
    using internal_iterator_type = std::vector<RegionIndex>::const_iterator;
    using value_type = RegionLayerHits;
    using difference_type = internal_iterator_type::difference_type;

    const_iterator(const IntermediateHitDoublets *ihd, internal_iterator_type iter): hitDoublets_(ihd), iter_(iter) {}
    value_type operator*() const {
      auto next = iter_+1;
      unsigned int end = hitDoublets_->layerPairs_.size();
      if(next != hitDoublets_->regions_.end())
        end = next->layerSetIndex();

      return RegionLayerHits(&(iter_->region()),
                             hitDoublets_->layerPairs_.begin() + iter_->layerSetIndex(),
                             hitDoublets_->layerPairs_.begin() + end);
    }

    const_iterator& operator++() { ++iter_; return *this; }
    const_iterator operator++(int) {
      const_iterator clone(*this);
      ++iter_;
      return clone;
    }

    bool operator==(const const_iterator& other) const { return iter_ == other.iter_; }
    bool operator!=(const const_iterator& other) const { return !operator==(other); }

  private:
    const IntermediateHitDoublets *hitDoublets_;
    internal_iterator_type iter_;
  };

  ////////////////////

  IntermediateHitDoublets(): seedingLayers_(nullptr) {}
  explicit IntermediateHitDoublets(const SeedingLayerSetsHits *seedingLayers): seedingLayers_(seedingLayers) {}
  IntermediateHitDoublets(const IntermediateHitDoublets& rh); // only to make ROOT dictionary generation happy
  ~IntermediateHitDoublets() = default;

  void swap(IntermediateHitDoublets& rh) {
    std::swap(seedingLayers_, rh.seedingLayers_);
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

  const SeedingLayerSetsHits& seedingLayerHits() const { return *seedingLayers_; }
  size_t regionSize() const { return regions_.size(); }

  const_iterator begin() const { return const_iterator(this, regions_.begin()); }
  const_iterator cbegin() const { return begin(); }
  const_iterator end() const { return const_iterator(this, regions_.end()); }
  const_iterator cend() const { return end(); }


private:
  const SeedingLayerSetsHits *seedingLayers_;

  std::vector<RegionIndex> regions_;
  std::vector<LayerPairHitDoublets> layerPairs_;
};

#endif
