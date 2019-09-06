#ifndef HeterogeneousCore_CUDATest_GPUTimeCruncher_h
#define HeterogeneousCore_CUDATest_GPUTimeCruncher_h

#include <cuda/api_wrappers.h>

#include <chrono>

namespace cudatest {
  /**
   * Calibrate the crunching finding the right relation between number
   * of iterations and time spent. The relation is linear.
   */
  class GPUTimeCruncher {
  public:
    GPUTimeCruncher();
    ~GPUTimeCruncher();

    GPUTimeCruncher(const GPUTimeCruncher&) = delete;
    GPUTimeCruncher& operator=(const GPUTimeCruncher&) = delete;
    GPUTimeCruncher(GPUTimeCruncher&&) = default;
    GPUTimeCruncher& operator=(GPUTimeCruncher&&) = default;

    void crunch_for(const std::chrono::nanoseconds& time, cuda::stream_t<>& stream) const;

  private:
    unsigned int getLoops(const std::chrono::nanoseconds& time) const;

    std::vector<unsigned int> niters_ = {
      0, 1, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048,
        2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192,
        9216, 10240, 12288, 14336, 16384, 20480, 28672, 32768,
        49152, 65536, 98304, 131072
    };
    std::vector<double> times_; // in us

    static constexpr size_t kernel_elements_ = 32;
    float* kernel_data_d_;
  };

  

  inline
  const GPUTimeCruncher& getGPUTimeCruncher() {
    const static GPUTimeCruncher cruncher;
    return cruncher;
  }

}

#endif
