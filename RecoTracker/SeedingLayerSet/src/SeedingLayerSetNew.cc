#include "RecoTracker/SeedingLayerSet/interface/SeedingLayerSetNew.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>

SeedingLayerSetNew::SeedingLayerSetNew(): SeedingLayerSetNew(0) {}
SeedingLayerSetNew::SeedingLayerSetNew(unsigned int nlayers): nlayers_(nlayers) {}
SeedingLayerSetNew::~SeedingLayerSetNew() {}

void SeedingLayerSetNew::addLayersHits(const std::vector<std::vector<ConstRecHitPointer> >& hits) {
  if(hits.size() != nlayers_)
    throw cms::Exception("Assert") << "Got " << hits.size() << " layers of hits, expected " << nlayers_;

  //std::cout << "Adding hits from " << hits.size() << " layers, nhits " << rechits_.size() << std::endl;

  for(const auto& layer: hits) {
    layerIndices_.push_back(rechits_.size());
    //std::cout << " adding " << layer.size() << " hits" << std::endl;
    std::copy(layer.begin(), layer.end(), std::back_inserter(rechits_));
  }
  //std::cout << "Finished, nhits " << rechits_.size() << std::endl;
}

std::vector<SeedingLayerSetNew::ConstRecHitPointer> SeedingLayerSetNew::hits(unsigned int layerSetIndex, unsigned int layerIndex) const {
  std::vector<ConstRecHitPointer> ret;
  unsigned int beginIndex = layerSetIndex*nlayers_ + layerIndex;
  unsigned int endIndex = beginIndex+1;

  unsigned int begin = layerIndices_[beginIndex];
  unsigned int end = rechits_.size();
  if(endIndex < layerIndices_.size())
    end = layerIndices_[endIndex];

  ret.reserve(end-begin);
  std::copy(rechits_.begin()+begin, rechits_.begin()+end, std::back_inserter(ret));
  return ret;
}
