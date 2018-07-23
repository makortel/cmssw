#ifndef HeterogeneousCore_CUDACore_TestCUDA_h
#define HeterogeneousCore_CUDACore_TestCUDA_h

#include "HeterogeneousCore/CUDACore/interface/CUDA.h"

/**
 * This class is intended only for testing purposes. It allows to
 * construct CUDA<T> and get the T from CUDA<T> without CUDAScopedContext.
 */
class TestCUDA {
public:
  template <typename T, typename ...Args>
  static CUDA<T> create(T data, Args&&... args) {
    return CUDA<T>(std::move(data), std::forward<Args>(args)...);
  }

  template <typename T>
  static const T& get(const CUDA<T>& data) {
    return data.data_;
  }
};

#endif
