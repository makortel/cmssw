#include "HeterogeneousCore/CUDACore/interface/CUDAToken.h"

namespace {
  auto make_stream(int device) {
    cuda::device::current::scoped_override_t<> setDeviceForThisScope(device);
    auto current_device = cuda::device::current::get();
    return std::make_unique<cuda::stream_t<>>(current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream));
  }
}

CUDAToken::CUDAToken(int device):
  stream_(make_stream(device)),
  device_(device)
{}

CUDAToken::~CUDAToken() {
  if(stream_) {
    // The current memory allocation model (large blocks) requires the
    // CUDA stream to be synchronized before moving on to the next
    // event in the EDM stream in order to avoid race conditions.
    stream_->synchronize();
  }
}
