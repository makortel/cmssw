#ifndef HeterogeneousCore_Producer_TestHeterogneousEDProducerGPUHelpers
#define HeterogeneousCore_Producer_TestHeterogneousEDProducerGPUHelpers

#include <cuda/api_wrappers.h>

#include <functional>
#include <memory>
#include <string>
#include <utility>

int TestHeterogeneousEDProducerGPUHelpers_simple_kernel(int input);

class TestHeterogeneousEDProducerGPUTask {
public:
  TestHeterogeneousEDProducerGPUTask() {}
  ~TestHeterogeneousEDProducerGPUTask() = default;

  using Ptr = cuda::memory::device::unique_ptr<float[]>;
  using PtrRaw = Ptr::pointer;
  
  using ResultType = std::pair<Ptr, Ptr>;
  using ResultTypeRaw = std::pair<PtrRaw, PtrRaw>;
  using ConstResultTypeRaw = std::pair<const PtrRaw, const PtrRaw>;

  using CallbackType = std::function<void(cuda::device::id_t, cuda::stream::id_t, cuda::status_t)>;

  ResultType runAlgo(const std::string& label, int input, const ResultTypeRaw inputArrays, CallbackType callback);
  void release(const std::string& label);
  static int getResult(const ResultTypeRaw& d_ac);

private:
  std::unique_ptr<cuda::stream_t<>> streamPtr;

  // temporary storage, need to be somewhere to allow async execution
  cuda::memory::host::unique_ptr<float[]> h_a;
  cuda::memory::host::unique_ptr<float[]> h_b;
  cuda::memory::device::unique_ptr<float[]> d_b;
  cuda::memory::device::unique_ptr<float[]> d_d;
  cuda::memory::device::unique_ptr<float[]> d_ma;
  cuda::memory::device::unique_ptr<float[]> d_mb;
  cuda::memory::device::unique_ptr<float[]> d_mc;
};

#endif
