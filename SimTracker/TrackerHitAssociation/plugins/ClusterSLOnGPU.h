// gpu
#include <cuda_runtime.h>
#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

#include "RecoLocalTracker/SiPixelClusterizer/plugins/siPixelRawToClusterHeterogeneousProduct.h"
#include "RecoLocalTracker/SiPixelRecHits/plugins/siPixelRecHitsHeterogeneousProduct.h"

struct ClusterSLGPU {
 ClusterSLGPU(){alloc();}
 void alloc();
 void zero(cudaStream_t stream);

 ClusterSLGPU * me_d;
 std::array<uint32_t,4> * links_d;
 uint32_t * tkId_d;
 uint32_t * tkId2_d;
 uint32_t * n1_d;
 uint32_t * n2_d;

 static constexpr uint32_t MAX_DIGIS = 2000*150;
 static constexpr uint32_t MaxNumModules = 2000;

};

namespace clusterSLOnGPU {

  using DigisOnGPU = siPixelRawToClusterHeterogeneousProduct::GPUProduct;
  using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
  using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;
  void wrapper(DigisOnGPU const & dd, uint32_t ndigis, HitsOnCPU const & hh, uint32_t nhits, ClusterSLGPU const & sl, uint32_t n, cuda::stream_t<>& stream);

}
