#ifndef TestGPU_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers
#define TestGPU_AcceleratorService_TestAcceleratorServiceProducerGPUHelpers

#include <cuda/api_wrappers.h>

#include <functional>
#include <memory>

int TestAcceleratorServiceProducerGPUHelpers_simple_kernel(int input);

class TestAcceleratorServiceProducerGPUTask {
public:
  TestAcceleratorServiceProducerGPUTask();
  ~TestAcceleratorServiceProducerGPUTask() = default;

  using ResultType = cuda::memory::device::unique_ptr<int[]>;
  using ResultTypeRaw = ResultType::pointer;
  
  ResultType runAlgo(int input, const ResultTypeRaw inputArray, std::function<void()> callback);
  int getResult(const ResultTypeRaw& d_c);

private:
  std::unique_ptr<cuda::stream_t<>> streamPtr;
};

#endif
