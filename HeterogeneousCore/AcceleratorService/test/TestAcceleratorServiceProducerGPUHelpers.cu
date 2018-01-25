#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "cuda/api_wrappers.h"
#include <cuda.h>
#include <cuda_runtime.h>

//
// Vector Addition Kernel
//
namespace {
template<typename T>
__global__
void vectorAdd(const T *a, const T *b, T *c, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numElements) { c[i] = a[i] + b[i]; }
}
}

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input) {
  // Example from Viktor/cuda-api-wrappers
  constexpr int NUM_VALUES = 10000;

  auto current_device = cuda::device::current::get();
  auto stream = current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

  auto h_a = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  auto h_b = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  auto h_c = cuda::memory::host::make_unique<int[]>(NUM_VALUES);

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = input + i;
    h_b[i] = i*i;
  }

  auto d_a = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_b = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);

  cuda::memory::async::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(int), stream.id());
  cuda::memory::async::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(int), stream.id());

  int threadsPerBlock {256};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  /*
    // doesn't work with header-only?
  cuda::launch(vectorAdd, {blocksPerGrid, threadsPerBlock},
               d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  */

  cuda::memory::async::copy(h_c.get(), d_c.get(), NUM_VALUES*sizeof(int), stream.id());

  stream.synchronize();

  int ret = 0;
  for (auto i=0; i<10; i++) {
    ret += h_c[i];
  }

  return ret;
}

namespace {
  constexpr int NUM_VALUES = 10000;
}

TestAcceleratorServiceProducerGPUTask::TestAcceleratorServiceProducerGPUTask() {
  auto current_device = cuda::device::current::get();
  streamPtr = std::make_unique<cuda::stream_t<>>(current_device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream));
}

TestAcceleratorServiceProducerGPUTask::ResultType
TestAcceleratorServiceProducerGPUTask::runAlgo(int input, const ResultTypeRaw inputArray, std::function<void()> callback) {
  auto h_a = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  auto h_b = cuda::memory::host::make_unique<int[]>(NUM_VALUES);

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i*i;
  }

  auto current_device = cuda::device::current::get();
  auto d_a = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_b = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  decltype(d_c) d_d;
  if(inputArray != nullptr) {
    d_d = cuda::memory::device::make_unique<int[]>(current_device, NUM_VALUES);
  }

  auto stream = *streamPtr;
  cuda::memory::async::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(int), stream.id());
  cuda::memory::async::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(int), stream.id());

  int threadsPerBlock {256};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  if(inputArray != nullptr) {
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(inputArray, d_c.get(), d_d.get(), NUM_VALUES);
    std::swap(d_c, d_d);
  }

  stream.enqueue.callback([callback](cuda::stream::id_t stream_id, cuda::status_t status){
      callback();
    });

  return d_c;
}

int TestAcceleratorServiceProducerGPUTask::getResult(const ResultTypeRaw& d_c) {
  auto h_c = cuda::memory::host::make_unique<int[]>(NUM_VALUES);
  cuda::memory::copy(h_c.get(), d_c, NUM_VALUES*sizeof(int));

  int ret = 0;
  for (auto i=0; i<10; i++) {
    ret += h_c[i];
  }

  return ret;
}
