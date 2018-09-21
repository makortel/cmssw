#ifndef CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h
#define CUDADataFormats_SiPixelCluster_interface_SiPixelClustersCUDA_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"

#include <cuda/api_wrappers.h>

class SiPixelClustersCUDA {
public:
  SiPixelClustersCUDA() = default;
  explicit SiPixelClustersCUDA(size_t feds, size_t nelements, cuda::stream_t<>& stream);
  ~SiPixelClustersCUDA() = default;

  SiPixelClustersCUDA(const SiPixelClustersCUDA&) = delete;
  SiPixelClustersCUDA& operator=(const SiPixelClustersCUDA&) = delete;
  SiPixelClustersCUDA(SiPixelClustersCUDA&&) = default;
  SiPixelClustersCUDA& operator=(SiPixelClustersCUDA&&) = default;

  uint32_t * __restrict__ moduleStart() { return moduleStart_d.get(); }
  int32_t  * __restrict__ clus() { return clus_d.get(); }
  uint32_t * __restrict__ clusInModule() { return clusInModule_d.get(); }
  uint32_t * __restrict__ moduleId() { return moduleId_d.get(); }
  uint32_t * __restrict__ clusModuleStart() { return clusModuleStart_d.get(); }

  uint32_t const * __restrict__ moduleStart() const { return moduleStart_d.get(); }
  int32_t  const * __restrict__ clus() const { return clus_d.get(); }
  uint32_t const * __restrict__ clusInModule() const { return clusInModule_d.get(); }
  uint32_t const * __restrict__ moduleId() const { return moduleId_d.get(); }
  uint32_t const * __restrict__ clusModuleStart() const { return clusModuleStart_d.get(); }

  uint32_t const * __restrict__ c_moduleStart() const { return moduleStart_d.get(); }
  int32_t  const * __restrict__ c_clus() const { return clus_d.get(); }
  uint32_t const * __restrict__ c_clusInModule() const { return clusInModule_d.get(); }
  uint32_t const * __restrict__ c_moduleId() const { return moduleId_d.get(); }
  uint32_t const * __restrict__ c_clusModuleStart() const { return clusModuleStart_d.get(); }

  struct DeviceConstView {
    uint32_t const *moduleStart;
    int32_t  const *clus;
    uint32_t const *clusInModule;
    uint32_t const *moduleId;
    uint32_t const *clusModuleStart;
  };

  DeviceConstView view() const { return DeviceConstView{moduleStart_d.get(), clus_d.get(), clusInModule_d.get(), moduleId_d.get(), clusModuleStart_d.get()}; }

private:
  edm::cuda::device::unique_ptr<uint32_t[]> moduleStart_d;   // index of the first pixel of each module
  edm::cuda::device::unique_ptr<int32_t[]>  clus_d;          // cluster id of each pixel
  edm::cuda::device::unique_ptr<uint32_t[]> clusInModule_d;  // number of clusters found in each module
  edm::cuda::device::unique_ptr<uint32_t[]> moduleId_d;      // module id of each module

  // originally from rechits
  edm::cuda::device::unique_ptr<uint32_t[]> clusModuleStart_d;
};

#endif
