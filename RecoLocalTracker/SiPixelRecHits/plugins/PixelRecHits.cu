// C++ headers
#include <algorithm>
#include <numeric>

// CUDA runtime
#include <cuda_runtime.h>

// CMSSW headers
#include "RecoLocalTracker/SiPixelClusterizer/plugins/SiPixelRawToClusterGPUKernel.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "RecoLocalTracker/SiPixelClusterizer/plugins/gpuClustering.h"
#include "PixelRecHits.h"
#include "gpuPixelRecHits.h"

namespace {
  __global__
  void setHitsLayerStart(const uint32_t* hitsModuleStart, const uint32_t* layerStart, uint32_t* hitsLayerStart) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < 10) {
      hitsLayerStart[i] = hitsModuleStart[layerStart[i]];
    }
    else if(i == 10) {
      hitsLayerStart[i] = hitsModuleStart[gpuClustering::MaxNumModules];
    }
  }
}

namespace pixelgpudetails {
  PixelRecHitGPUKernel::PixelRecHitGPUKernel(cuda::stream_t<>& cudaStream) {

    constexpr auto MAX_HITS = gpuClustering::MaxNumModules * 256;

    cudaCheck(cudaMalloc((void **) & gpu_.bs_d, 3 * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.hitsLayerStart_d, 11 * sizeof(uint32_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.charge_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.detInd_d, MAX_HITS * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.xg_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.yg_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.zg_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.rg_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.xl_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.yl_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.xerr_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.yerr_d, MAX_HITS * sizeof(float)));
    cudaCheck(cudaMalloc((void **) & gpu_.iphi_d, MAX_HITS * sizeof(int16_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.sortIndex_d, MAX_HITS * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.mr_d, MAX_HITS * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.mc_d, MAX_HITS * sizeof(uint16_t)));
    cudaCheck(cudaMalloc((void **) & gpu_.hist_d, 10 * sizeof(HitsOnGPU::Hist)));
    cudaCheck(cudaMalloc((void **) & gpu_d, sizeof(HitsOnGPU)));
    gpu_.me_d = gpu_d;
    cudaCheck(cudaMemcpyAsync(gpu_d, &gpu_, sizeof(HitsOnGPU), cudaMemcpyDefault, cudaStream.id()));

    // Feels a bit dumb but constexpr arrays are not supported for device code
    // TODO: should be moved to EventSetup (or better ideas?)
    // Would it be better to use "constant memory"?
    cudaCheck(cudaMalloc((void **) & d_phase1TopologyLayerStart_, 11 * sizeof(uint32_t)));
    cudaCheck(cudaMemcpyAsync(d_phase1TopologyLayerStart_, phase1PixelTopology::layerStart, 11 * sizeof(uint32_t), cudaMemcpyDefault, cudaStream.id()));

    cudaCheck(cudaMallocHost(&h_hitsModuleStart_, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t)));
    cudaCheck(cudaMallocHost(&h_detInd_, MAX_HITS*sizeof(uint16_t)));
    cudaCheck(cudaMallocHost(&h_charge_, MAX_HITS*sizeof(int32_t)));
    cudaCheck(cudaMallocHost(&h_xl_, MAX_HITS*sizeof(float)));
    cudaCheck(cudaMallocHost(&h_yl_, MAX_HITS*sizeof(float)));
    cudaCheck(cudaMallocHost(&h_xe_, MAX_HITS*sizeof(float)));
    cudaCheck(cudaMallocHost(&h_ye_, MAX_HITS*sizeof(float)));
    cudaCheck(cudaMallocHost(&h_mr_, MAX_HITS*sizeof(uint16_t)));
    cudaCheck(cudaMallocHost(&h_mc_, MAX_HITS*sizeof(uint16_t)));
#ifdef GPU_DEBUG
    cudaCheck(cudaMallocHost(&h_hitsLayerStart_, 11 * sizeof(uint32_t)));
#endif
  }
  PixelRecHitGPUKernel::~PixelRecHitGPUKernel() {
    cudaCheck(cudaFree(gpu_.bs_d));
    cudaCheck(cudaFree(gpu_.hitsLayerStart_d));
    cudaCheck(cudaFree(gpu_.charge_d));
    cudaCheck(cudaFree(gpu_.detInd_d));
    cudaCheck(cudaFree(gpu_.xg_d));
    cudaCheck(cudaFree(gpu_.yg_d));
    cudaCheck(cudaFree(gpu_.zg_d));
    cudaCheck(cudaFree(gpu_.rg_d));
    cudaCheck(cudaFree(gpu_.xl_d));
    cudaCheck(cudaFree(gpu_.yl_d));
    cudaCheck(cudaFree(gpu_.xerr_d));
    cudaCheck(cudaFree(gpu_.yerr_d));
    cudaCheck(cudaFree(gpu_.iphi_d));
    cudaCheck(cudaFree(gpu_.sortIndex_d));
    cudaCheck(cudaFree(gpu_.mr_d));
    cudaCheck(cudaFree(gpu_.mc_d));
    cudaCheck(cudaFree(gpu_.hist_d));
    cudaCheck(cudaFree(gpu_d));
    cudaCheck(cudaFree(d_phase1TopologyLayerStart_));

    cudaCheck(cudaFreeHost(h_hitsModuleStart_));
    cudaCheck(cudaFreeHost(h_detInd_));
    cudaCheck(cudaFreeHost(h_charge_));
    cudaCheck(cudaFreeHost(h_xl_));
    cudaCheck(cudaFreeHost(h_yl_));
    cudaCheck(cudaFreeHost(h_xe_));
    cudaCheck(cudaFreeHost(h_ye_));
    cudaCheck(cudaFreeHost(h_mr_));
    cudaCheck(cudaFreeHost(h_mc_));
#ifdef GPU_DEBUG
    cudaCheck(cudaFreeHost(h_hitsLayerStart_));
#endif
  }

  void PixelRecHitGPUKernel::makeHitsAsync(const siPixelRawToClusterHeterogeneousProduct::GPUProduct& input,
                                           float const * bs,
                                           pixelCPEforGPU::ParamsOnGPU const * cpeParams,
                                           cuda::stream_t<>& stream) {
    cudaCheck(cudaMemcpyAsync(gpu_.bs_d, bs, 3 * sizeof(float), cudaMemcpyDefault, stream.id()));
    gpu_.hitsModuleStart_d = input.clusModuleStart_d;
    cudaCheck(cudaMemcpyAsync(gpu_d, &gpu_, sizeof(HitsOnGPU), cudaMemcpyDefault, stream.id()));

    int threadsPerBlock = 256;
    int blocks = input.nModules; // active modules (with digis)
    gpuPixelRecHits::getHits<<<blocks, threadsPerBlock, 0, stream.id()>>>(
      cpeParams,
      gpu_.bs_d,
      input.moduleInd_d,
      input.xx_d, input.yy_d, input.adc_d,
      input.moduleStart_d,
      input.clusInModule_d, input.moduleId_d,
      input.clus_d,
      input.nDigis,
      gpu_.hitsModuleStart_d,
      gpu_.charge_d,
      gpu_.detInd_d,
      gpu_.xg_d, gpu_.yg_d, gpu_.zg_d, gpu_.rg_d,
      gpu_.iphi_d,
      gpu_.xl_d, gpu_.yl_d,
      gpu_.xerr_d, gpu_.yerr_d,
      gpu_.mr_d, gpu_.mc_d
    );
    cudaCheck(cudaGetLastError());

    // assuming full warp of threads is better than a smaller number...
    setHitsLayerStart<<<1, 32, 0, stream.id()>>>(gpu_.hitsModuleStart_d, d_phase1TopologyLayerStart_, gpu_.hitsLayerStart_d);
    cudaCheck(cudaGetLastError());

    // needed only if hits on CPU are required...
    nhits_ = input.nClusters;
    cudaCheck(cudaMemcpyAsync(h_hitsModuleStart_, gpu_.hitsModuleStart_d, (gpuClustering::MaxNumModules+1) * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
#ifdef GPU_DEBUG
    cudaCheck(cudaMemcpyAsync(h_hitsLayerStart_, gpu_.hitsLayerStart_d, 11 * sizeof(uint32_t), cudaMemcpyDefault, stream.id()));
#endif
    cudaCheck(cudaMemcpyAsync(h_detInd_, gpu_.detInd_d, nhits_*sizeof(int16_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_charge_, gpu_.charge_d, nhits_ * sizeof(int32_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_xl_, gpu_.xl_d, nhits_ * sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_yl_, gpu_.yl_d, nhits_ * sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_xe_, gpu_.xerr_d, nhits_ * sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_ye_, gpu_.yerr_d, nhits_ * sizeof(float), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_mr_, gpu_.mr_d, nhits_ * sizeof(uint16_t), cudaMemcpyDefault, stream.id()));
    cudaCheck(cudaMemcpyAsync(h_mc_, gpu_.mc_d, nhits_ * sizeof(uint16_t), cudaMemcpyDefault, stream.id()));

#ifdef GPU_DEBUG
    cudaStreamSynchronize(stream.id());

    std::cout << "hit layerStart ";
    for (int i=0;i<10;++i) std::cout << phase1PixelTopology::layerName[i] << ':' << h_hitsLayerStart_[i] << ' ';
    std::cout << "end:" << h_hitsLayerStart_[10] << std::endl;
#endif

    // for timing test
    // cudaStreamSynchronize(stream.id());
    // auto nhits_ = h_hitsLayerStart_[10];
    // radixSortMultiWrapper<int16_t><<<10, 256, 0, c.stream>>>(gpu_.iphi_d, gpu_.sortIndex_d, gpu_.hitsLayerStart_d);

    cudautils::fillManyFromVector(gpu_.hist_d, 10, gpu_.iphi_d, gpu_.hitsLayerStart_d, nhits_, 256, stream.id());
  }
}
