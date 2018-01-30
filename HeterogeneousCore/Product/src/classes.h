#include "DataFormats/Common/interface/Wrapper.h"
#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"
#include <utility>

namespace {
  struct dictionary {
    HeterogeneousProduct<heterogeneous::CPUProduct<unsigned int>, heterogeneous::GPUMockProduct<unsigned int>> hpuu;
    HeterogeneousProduct<heterogeneous::CPUProduct<unsigned int>, heterogeneous::GPUCudaProduct<int*>> hpuip;
    HeterogeneousProduct<heterogeneous::CPUProduct<unsigned int>, heterogeneous::GPUCudaProduct<float*>> hpufp;
    HeterogeneousProduct<heterogeneous::CPUProduct<unsigned int>, heterogeneous::GPUCudaProduct<std::pair<float*, float*>>> hpupfpfp;
  };
}
