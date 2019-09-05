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

    auto h_src = cudautils::make_host_noncached_unique<char[]>(kernel_elements_*sizeof(float) /*, cudaHostAllocWriteCombined*/);
    cuda::throw_if_error(cudaMalloc(&kernel_data_d_, kernel_elements_*sizeof(float)));
    for(size_t i=0; i!=kernel_elements_; ++i) {
      h_src[i] = dis(gen);
    }
    cuda::throw_if_error(cudaMemcpy(kernel_data_d_, h_src.get(), kernel_elements_*sizeof(float), cudaMemcpyDefault));

    auto streamPtr = cudautils::getCUDAStreamCache().getCUDAStream();
    cudaEvent_t start, stop;
    cuda::throw_if_error(cudaEventCreate(&start));
    cuda::throw_if_error(cudaEventCreate(&stop));

    // calibrate
    times_.reserve(niters_.size());
    for(auto n: niters_) {
      cuda::throw_if_error(cudaEventRecord(start, streamPtr->id()));
      TestCUDAProducerSimEWGPUKernel::kernel(kernel_data_d_, kernel_elements_, n, *streamPtr);
      cuda::throw_if_error(cudaEventRecord(stop, streamPtr->id()));
      cuda::throw_if_error(cudaEventSynchronize(stop));
      float ms;
      cuda::throw_if_error(cudaEventElapsedTime(&ms, start, stop));
      
      edm::LogPrint("foo") << "Crunched " << n << " iterations for " << ms*1000 << " us" << std::endl;

      times_.push_back(ms*1000); // convert to us
    }


    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  GPUTimeCruncher::~GPUTimeCruncher() {
    cuda::throw_if_error(cudaFree(kernel_data_d_));
  }

  void GPUTimeCruncher::crunch_for(const std::chrono::microseconds& time, cuda::stream_t<>& stream) const {
    const auto loops = getLoops(time);
    TestCUDAProducerSimEWGPUKernel::kernel(kernel_data_d_, kernel_elements_, loops, stream);
  }

  unsigned int GPUTimeCruncher::getLoops(const std::chrono::microseconds& time) const {
    const float runtime = time.count();
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
