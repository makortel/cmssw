#ifndef HeterogeneousCore_CUDATest_CUDAThing_H
#define HeterogeneousCore_CUDATest_CUDAThing_H

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"

class CUDAThing {
public:
  CUDAThing() = default;
  CUDAThing(edm::cuda::device::unique_ptr<float[]> ptr):
    ptr_(std::move(ptr))
  {}

  const float *get() const { return ptr_.get(); }

private:
  edm::cuda::device::unique_ptr<float[]> ptr_;;
};

#endif
