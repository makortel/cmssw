#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"
#include "FWCore/Utilities/interface/Exception.h"

SiPixelClusterShapeData::~SiPixelClusterShapeData() {}

SiPixelClusterShapeCache::LazyGetter::LazyGetter() {}
SiPixelClusterShapeCache::LazyGetter::~LazyGetter() {}

SiPixelClusterShapeCache::~SiPixelClusterShapeCache() {}

void SiPixelClusterShapeCache::check_productId(edm::ProductID id) const {
  if(id != productId_)
    throw cms::Exception("InvalidReference") << "SiPixelClusterShapeCache caches values for SiPixelClusters with ProductID " << productId_ << ", got SiPixelClusterRef with ID " << id;
}

void SiPixelClusterShapeCache::check_index(ClusterRef::key_type index) const {
  if(index >= data_.size())
    throw cms::Exception("InvalidReference") << "SiPixelClusterShapeCache caches values for SiPixelClusters with ProductID " << productId_ << " that has " << data_.size() << " clusters, got SiPixelClusterRef with index " << index;
}
