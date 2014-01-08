#include "RecoTracker/SeedingLayerSet/interface/SeedingLayerSetNew.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <limits>

SeedingLayerSetNew::SeedingLayerSetNew(): SeedingLayerSetNew(0) {}
SeedingLayerSetNew::SeedingLayerSetNew(unsigned int nlayers): nlayers_(nlayers) {}
SeedingLayerSetNew::~SeedingLayerSetNew() {}


std::pair<unsigned int, bool> SeedingLayerSetNew::insertLayer(const std::string& layerName, const DetLayer *layerDet) {
  std::pair<unsigned int, bool> index = insertLayer_(layerName, layerDet);
  layersIndices_.push_back(index.first);
  return index;
}

std::pair<unsigned int, bool> SeedingLayerSetNew::insertLayer_(const std::string& layerName, const DetLayer *layerDet) {
  auto found = std::find(layerNames_.begin(), layerNames_.end(), layerName);
  // insert if not found
  if(found == layerNames_.end()) {
    layerNames_.push_back(layerName);
    layerDets_.push_back(layerDet);
    const auto max = std::numeric_limits<unsigned int>::max();
    layerHitRanges_.emplace_back(max, max);
    //std::cout << "Inserted layer " << layerName << " to index " << layerNames_.size()-1 << std::endl;
    return std::make_pair(layerNames_.size()-1, true);
  }
  //std::cout << "Encountered layer " << layerName << " index " << found-layerNames_.begin() << std::endl;
  return std::make_pair(found-layerNames_.begin(), false);
}

void SeedingLayerSetNew::insertLayerHits(unsigned int layerIndex, const Hits& hits) {
  assert(layerIndex < layerHitRanges_.size());
  Range& range = layerHitRanges_[layerIndex];
  range.first = rechits_.size();
  std::copy(hits.begin(), hits.end(), std::back_inserter(rechits_));
  range.second = rechits_.size();

  //std::cout << "  added " << hits.size() << " hits to layer " << layerIndex << " range " << range.first << " " << range.second << std::endl;
}

SeedingLayerSetNew::Hits SeedingLayerSetNew::hits(unsigned int layerIndex) const {
  const Range& range = layerHitRanges_[layerIndex];

  Hits ret;
  ret.reserve(range.second-range.first);
  std::copy(rechits_.begin()+range.first, rechits_.begin()+range.second, std::back_inserter(ret));
  return ret;
}


void SeedingLayerSetNew::print() const {
  std::cout << "SeedingLayerSetNew with " << sizeLayers() << " layers in each set, layer set has " << sizeLayerSets() << " items" << std::endl;
  for(unsigned iLayers=0; iLayers<sizeLayerSets(); ++iLayers) {
    std::cout << " " << iLayers << ": ";
    SeedingLayers layers = getLayers(iLayers);
    for(unsigned iLayer=0; iLayer<layers.size(); ++iLayer) {
      SeedingLayer layer = layers.getLayer(iLayer);
      std::cout << layer.name() << " (" << layer.index() << ", nhits " << layer.hits().size() << ") ";
    }
    std::cout << std::endl;
  }
}
