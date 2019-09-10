// -*- C++ -*-

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAStreamCache.h"

#include "GPUTimeCruncher.h"
#include "TestCUDAProducerSimEWGPUKernel.h"

#include <cuda.h>

#include <random>

namespace cudatest {
  GPUTimeCruncher::GPUTimeCruncher() {
    // Data for kernel
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(1e-5, 100.);

    float *kernel_data_d;
    auto h_src = cudautils::make_host_noncached_unique<char[]>(kernel_elements*sizeof(float) /*, cudaHostAllocWriteCombined*/);
    cuda::throw_if_error(cudaMalloc(&kernel_data_d, kernel_elements*sizeof(float)));
    for(size_t i=0; i!=kernel_elements; ++i) {
      h_src[i] = dis(gen);
    }
    cuda::throw_if_error(cudaMemcpy(kernel_data_d, h_src.get(), kernel_elements*sizeof(float), cudaMemcpyDefault));


    // calibrate
    times_.reserve(niters_.size());
#ifdef CALIBRATE_EVENT
    auto streamPtr = cudautils::getCUDAStreamCache().getCUDAStream();
    cudaEvent_t start, stop;
    cuda::throw_if_error(cudaEventCreate(&start));
    cuda::throw_if_error(cudaEventCreate(&stop));
    for(auto n: niters_) {
      cuda::throw_if_error(cudaEventRecord(start, streamPtr->id()));
      TestCUDAProducerSimEWGPUKernel::kernel(kernel_data_d, kernel_elements, n, *streamPtr);
      cuda::throw_if_error(cudaEventRecord(stop, streamPtr->id()));
      cuda::throw_if_error(cudaEventSynchronize(stop));
      float ms;
      cuda::throw_if_error(cudaEventElapsedTime(&ms, start, stop));
      
      edm::LogPrint("foo") << "Crunched " << n << " iterations for " << ms*1000 << " us" << std::endl;

      times_.push_back(ms*1000); // convert to us
    }
    times_[0] = 0.;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif
    // Calibration with events seems to be too imprecise (I get
    // smallest kernel runtime of 10us, while nvvp shows 2 us), so
    // let's do it by hand...
    times_ = {2.144,
              3.168,
              3.616,
              5.823,
              10.176,
              18.88,
              36.255,
              53.76,
              71.136,
              88.447,
              105.823,
              123.135,
              140.479,
              175.134,
              209.983,
              244.607,
              279.614,
              314.334,
              349.053,
              418.269,
              487.932,
              557.692,
              696.635,
              973.946,
              1112.09,
              1667.96,
              2223.73,
              3335.66,
              4447.62};
    assert(times_.size() == niters_.size());
  }

  void GPUTimeCruncher::crunch_for(const std::chrono::nanoseconds& time, float* kernel_data_d, cuda::stream_t<>& stream) const {
    const auto loops = getLoops(time);
    TestCUDAProducerSimEWGPUKernel::kernel(kernel_data_d, kernel_elements, loops, stream);
  }

  unsigned int GPUTimeCruncher::getLoops(const std::chrono::nanoseconds& time) const {
    const double runtime = time.count()/1000.;
    bool found;
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
                    << "  loops: " << loops;
    return loops;

  }
}
