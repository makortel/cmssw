#ifndef HeterogeneousCore_CUDATest_SimOperations_h
#define HeterogeneousCore_CUDATest_SimOperations_h

#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include <cuda/api_wrappers.h>

#include <memory>
#include <string>
#include <vector>

namespace cudatest {
  class OperationBase;
  class GPUTimeCruncher;

  class SimOperations {
  public:
    explicit SimOperations(const std::string& configFile,
                           const std::string& cudaCalibrationFile,
                           const std::string& nodepath,
                           const unsigned int gangSize = 1,
                           const double gangKernelFactor = 1.0);
    ~SimOperations();

    SimOperations(const SimOperations&) = delete;
    SimOperations& operator=(const SimOperations&) = delete;
    SimOperations(SimOperations&&) = default;
    SimOperations& operator=(SimOperations&&) = default;

    size_t events() const;

    void operate(const std::vector<size_t>& indices, cuda::stream_t<>* stream);

  private:
    using OpVector = std::vector<std::unique_ptr<OperationBase>>;
    OpVector ops_;

    std::unique_ptr<GPUTimeCruncher> gpuCruncher_;
    float* kernel_data_d_;

    // These are indexed by the operation index in ops_
    // They are likely to contain null elements
    std::vector<char* > data_d_src_;
    std::vector<cudautils::host::noncached::unique_ptr<char[]>> data_h_src_;
  };
}

#endif
