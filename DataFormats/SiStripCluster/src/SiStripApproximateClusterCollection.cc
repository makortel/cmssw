#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"

void SiStripApproximateClusterCollection::reserve(size_t dets, size_t clusters) {
  detIds_.reserve(dets);
  offsetsToEnd_.reserve(dets);
  clusters_.reserve(clusters);
}

SiStripApproximateClusterCollection::Filler SiStripApproximateClusterCollection::beginDet(unsigned int detId) {
  detIds_.push_back(detId);
  offsetsToEnd_.push_back(0);  // updated in Filler()
  return Filler(clusters_, offsetsToEnd_.back());
}
