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
