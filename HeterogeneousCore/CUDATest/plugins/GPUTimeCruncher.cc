// -*- C++ -*-

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include "GPUTimeCruncher.h"
#include "TestCUDAProducerSimEWGPUKernel.h"

#include <cuda.h>

#include <random>

namespace cudatest {
  GPUTimeCruncher::GPUTimeCruncher(const std::vector<unsigned int>& iters, const std::vector<double>& times) {
    if(iters.size() != times.size()) {
      throw cms::Exception("Configuration") << "CUDA Calibration: got " << iters.size() << " iterations and " << times.size() << " times";
    }
    if(iters.empty()) {
      throw cms::Exception("Configuration") << "CUDA Calibration: iterations is empty";
    }
    if(times.empty()) {
      throw cms::Exception("Configuration") << "CUDA Calibration: times is empty";
    }
    niters_ = iters;
    times_ = times;

    // Data for kernel
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);

    float *kernel_data_d;
    auto h_src = cms::cuda::make_host_noncached_unique<char[]>(kernel_elements*sizeof(float) /*, cudaHostAllocWriteCombined*/);
    cudaCheck(cudaMalloc(&kernel_data_d, kernel_elements*sizeof(float)));
    for(size_t i=0; i!=kernel_elements; ++i) {
      h_src[i] = dis(gen);
    }
    cudaCheck(cudaMemcpy(kernel_data_d, h_src.get(), kernel_elements*sizeof(float), cudaMemcpyDefault));
  }

  void GPUTimeCruncher::crunch_for(const std::chrono::nanoseconds& time, float* kernel_data_d, cudaStream_t stream) const {
    const auto loops = getLoops(time);
    TestCUDAProducerSimEWGPUKernel::kernel(kernel_data_d, kernel_elements, loops, stream);
  }

  unsigned int GPUTimeCruncher::getLoops(const std::chrono::nanoseconds& time) const {
    const double runtime = time.count()/1000.;
    bool found = false;
    size_t smaller_i = 0;
    for(size_t i=1; i<times_.size(); ++i) {
      if(times_[i] > runtime) {
        smaller_i = i;
        found = true;
        break;
      }
    }

    if(not found) {
      smaller_i = times_.size()-2;
    }
    const auto x0 = times_[smaller_i];
    const auto x1 = times_[smaller_i+1];
    const auto y0 = niters_[smaller_i];
    const auto y1 = niters_[smaller_i+1];
    const double m = double(y1-y0) / double(x1-x0);
    const double q = y0 - m*x0;
    const unsigned int loops = m*runtime + 1;
    LogDebug("foo") << "x0: " << x0 << " x1: " << x1 << " y0: " << y0 << " y1: " << y1 << "  m: " << m << " q: " << q
                    << "  loops: " << loops << " asked for " << runtime << " us";
    return loops;

  }
}
