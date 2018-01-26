#include "TestAcceleratorServiceProducerGPUHelpers.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cuda/api_wrappers.h>
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

  template <typename T>
  __global__
  void vectorProd(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < numElements && col < numElements) {
      c[row*numElements + col] = a[row]*b[col];
    }
  }

  template <typename T>
  __global__
  void matrixMul(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if(row < numElements && col < numElements) {
      T tmp = 0;
      for(int i=0; i<numElements; ++i) {
        tmp += a[row*numElements + i] * b[i*numElements + col];
      }
      c[row*numElements + col] = tmp;
    }
  }

  template <typename T>
  __global__
  void matrixMulVector(const T *a, const T *b, T *c, int numElements) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;

    if(row < numElements) {
      T tmp = 0;
      for(int i=0; i<numElements; ++i) {
        tmp += a[row*numElements + i] * b[i];
      }
      c[row] = tmp;
    }
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
  h_a = cuda::memory::host::make_unique<float[]>(NUM_VALUES);
  h_b = cuda::memory::host::make_unique<float[]>(NUM_VALUES);

  for (auto i=0; i<NUM_VALUES; i++) {
    h_a[i] = i;
    h_b[i] = i*i;
  }

  auto current_device = cuda::device::current::get();
  d_a = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  d_b = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  auto d_c = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  if(inputArray != nullptr) {
    d_d = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES);
  }

  d_ma = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);
  d_mb = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);
  d_mc = cuda::memory::device::make_unique<float[]>(current_device, NUM_VALUES*NUM_VALUES);

  auto& stream = *streamPtr;
  cuda::memory::async::copy(d_a.get(), h_a.get(), NUM_VALUES*sizeof(float), stream.id());
  cuda::memory::async::copy(d_b.get(), h_b.get(), NUM_VALUES*sizeof(float), stream.id());

  int threadsPerBlock {32};
  int blocksPerGrid = (NUM_VALUES + threadsPerBlock - 1) / threadsPerBlock;

  edm::LogPrint("Foo") << "--- launching kernels";
  vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_a.get(), d_b.get(), d_c.get(), NUM_VALUES);
  if(inputArray != nullptr) {
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(inputArray, d_c.get(), d_d.get(), NUM_VALUES);
    std::swap(d_c, d_d);
  }

  dim3 threadsPerBlock3{NUM_VALUES, NUM_VALUES};
  dim3 blocksPerGrid3{1,1};
  if(NUM_VALUES*NUM_VALUES > 32) {
    threadsPerBlock3.x = 32;
    threadsPerBlock3.y = 32;
    blocksPerGrid3.x = ceil(double(NUM_VALUES)/double(threadsPerBlock3.x));
    blocksPerGrid3.y = ceil(double(NUM_VALUES)/double(threadsPerBlock3.y));
  }
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_a.get(), d_b.get(), d_ma.get(), NUM_VALUES);
  vectorProd<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_a.get(), d_c.get(), d_mb.get(), NUM_VALUES);
  matrixMul<<<blocksPerGrid3, threadsPerBlock3, 0, stream.id()>>>(d_ma.get(), d_mb.get(), d_mc.get(), NUM_VALUES);

  matrixMulVector<<<blocksPerGrid, threadsPerBlock, 0, stream.id()>>>(d_mc.get(), d_b.get(), d_c.get(), NUM_VALUES);

  edm::LogPrint("Foo") << "--- kernels launched, enqueueing the callback";
  stream.enqueue.callback([callback](cuda::stream::id_t stream_id, cuda::status_t status){
      callback();
    });

  edm::LogPrint("Foo") << "--- finished, returning return pointer";
  return d_c;
}

void TestAcceleratorServiceProducerGPUTask::release() {
  // any way to automate the release?
  edm::LogPrint("Foo") << "--- releasing temporary memory";
  h_a.reset();
  h_b.reset();
  d_a.reset();
  d_b.reset();
  d_d.reset();
  d_ma.reset();
  d_mb.reset();
  d_mc.reset();
}

int TestAcceleratorServiceProducerGPUTask::getResult(const ResultTypeRaw& d_c) {
  auto h_c = cuda::memory::host::make_unique<float[]>(NUM_VALUES);
  cuda::memory::copy(h_c.get(), d_c, NUM_VALUES*sizeof(int));

  float ret = 0;
  for (auto i=0; i<NUM_VALUES; i++) {
    ret += h_c[i];
  }

  return static_cast<int>(ret);
}

