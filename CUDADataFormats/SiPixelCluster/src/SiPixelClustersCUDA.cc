#include "CUDADataFormats/SiPixelCluster/interface/SiPixelClustersCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

SiPixelClustersCUDA::SiPixelClustersCUDA(size_t feds, size_t nelements, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  moduleStart_d     = cs->make_device_unique<uint32_t[]>(nelements+1, stream);
  clus_d            = cs->make_device_unique< int32_t[]>(feds, stream);
  clusInModule_d    = cs->make_device_unique<uint32_t[]>(nelements, stream);
  moduleId_d        = cs->make_device_unique<uint32_t[]>(nelements, stream);
  clusModuleStart_d = cs->make_device_unique<uint32_t[]>(nelements+1, stream);
}
