#include "HeterogeneousCore/CUDACore/interface/CUDAESManaged.h"


CUDAESManaged::CUDAESManaged() {}

CUDAESManaged::~CUDAESManaged() {
  for(auto& ptrSize: buffers_) {
    cudaFree(ptrSize.first);
  }
}

void CUDAESManaged::advise() const {
  for(const auto& ptrSize: buffers_) {
    cudaCheck(cudaMemAdvise(ptrSize.first, ptrSize.second, cudaMemAdviseSetReadMostly, 0)); // device is ignored for this advise
  }
}

void CUDAESManaged::prefetchAsync(cuda::stream_t<>& stream) const {
  for(const auto& ptrSize: buffers_) {
    cudaMemPrefetchAsync(ptrSize.first, ptrSize.second, stream.device_id(), stream.id());
  }
}
