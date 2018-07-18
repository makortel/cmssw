#ifndef HeterogeneousCore_CUDACore_TestCUDA_h
#define HeterogeneousCore_CUDACore_TestCUDA_h

class TestCUDA {
public:
  template <typename T, typename ...Args>
  static CUDA<T> create(T data, Args&&... args) {
    return CUDA<T>(std::move(data), std::forward<Args>(args)...);
  }
};

#endif
