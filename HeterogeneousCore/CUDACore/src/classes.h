#include "DataFormats/Common/interface/Wrapper.h"
#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"
#include "HeterogeneousCore/CUDACore/interface/CUDA.h"

namespace {
  struct dictionary {
    CUDAToken ct;

    // These should really be placed elsewhere?
    CUDA<float *> cf;
  };
}
