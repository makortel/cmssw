#ifndef TestGPU_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers
#define TestGPU_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers

#include <cuda/api_wrappers.h>

#include <memory>

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input);

class TestAcceleratorServiceProducerGPUTask {
public:
  TestAcceleratorServiceProducerGPUTask();
  ~TestAcceleratorServiceProducerGPUTask() = default;

  void runAlgo(int input);
  int getResult();

private:
  std::unique_ptr<cuda::stream_t<>> streamPtr;
  cuda::memory::host::unique_ptr<int[]> h_c;
  cuda::memory::device::unique_ptr<int[]> d_c;
};

#endif
