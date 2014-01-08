#ifndef RecoTracker_SeedingLayerSet_SeedingLayerSetNew
#define RecoTracker_SeedingLayerSet_SeedingLayerSetNew

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include <vector>
#include <string>
#include <utility>

class SeedingLayerSetNew {
public:
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer> Hits;

  class SeedingLayer {
  public:
    SeedingLayer(): seedingLayerSets_(0), index_(0) {}
    SeedingLayer(const SeedingLayerSetNew *sls, unsigned int index):
      seedingLayerSets_(sls), index_(index) {}

    unsigned int index() const { return index_; }
    const std::string& name() const { return seedingLayerSets_->layerNames_[index_]; }
    Hits hits() const { return seedingLayerSets_->hits(index_); }

  private:
    const SeedingLayerSetNew *seedingLayerSets_;
    unsigned int index_;
  };

  class SeedingLayers {
  public:
    SeedingLayers(): seedingLayerSets_(0) {}
    SeedingLayers(const SeedingLayerSetNew *sls, std::vector<unsigned int>::const_iterator begin, std::vector<unsigned int>::const_iterator end):
      seedingLayerSets_(sls), begin_(begin), end_(end) {}

    unsigned int size() const { return end_-begin_; }
    SeedingLayer getLayer(unsigned int index) const {
      return SeedingLayer(seedingLayerSets_, *(begin_+index));
    }

  private:
    const SeedingLayerSetNew *seedingLayerSets_;
    std::vector<unsigned int>::const_iterator begin_;
    std::vector<unsigned int>::const_iterator end_;
  };

  SeedingLayerSetNew();
  explicit SeedingLayerSetNew(unsigned int nlayers_);
  ~SeedingLayerSetNew();

  std::pair<unsigned int, bool> insertLayer(const std::string& layerName);
  void insertLayerHits(unsigned int layerIndex, const Hits& hits);


  unsigned int sizeLayers() const { return nlayers_; }
  unsigned int sizeLayerSets() const { return layersIndices_.size() / nlayers_; }

  SeedingLayers getLayers(unsigned int index) const {
    std::vector<unsigned int>::const_iterator begin = layersIndices_.begin()+nlayers_*index;
    std::vector<unsigned int>::const_iterator end = begin+nlayers_;
    return SeedingLayers(this, begin, end);
  }

  void print() const;

private:
  std::pair<unsigned int, bool> insertLayer_(const std::string& layerName);
  Hits hits(unsigned int layerIndex) const;

  unsigned int nlayers_; // number of layers in a SeedingLayers

  std::vector<unsigned int> layersIndices_;

  // index is the layer index
  typedef std::pair<unsigned int, unsigned int> Range;
  std::vector<Range> layerHitRanges_; // maps index of a layer to an index in rechits_ for the start of the hit list
  std::vector<std::string> layerNames_;

  // index from layerIndices_
  std::vector<ConstRecHitPointer> rechits_; // list of rechits
};

#endif
