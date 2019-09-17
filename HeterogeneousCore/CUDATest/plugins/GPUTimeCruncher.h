#ifndef HeterogeneousCore_CUDATest_GPUTimeCruncher_h
#define HeterogeneousCore_CUDATest_GPUTimeCruncher_h

#include <cuda/api_wrappers.h>

#include <chrono>
#include <vector>

namespace cudatest {
  /**
   * Calibrate the crunching finding the right relation between number
   * of iterations and time spent. The relation is linear.
   */
  class GPUTimeCruncher {
  public:
    GPUTimeCruncher();

    void setCalibration(const std::vector<unsigned int>& iters, const std::vector<double>& times) const;

    void crunch_for(const std::chrono::nanoseconds& time, float* kernel_data_d, cuda::stream_t<>& stream) const;

    static constexpr size_t kernel_elements = 32;

  private:
    unsigned int getLoops(const std::chrono::nanoseconds& time) const;

    std::vector<unsigned int> niters_;
    std::vector<double> times_; // in us
  };

  

  inline
  const GPUTimeCruncher& getGPUTimeCruncher() {
    const static GPUTimeCruncher cruncher;
    return cruncher;
  }

}

#endif
