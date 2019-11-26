#ifndef HeterogeneousCore_CUDAUtilities_cudaCheck_h
#define HeterogeneousCore_CUDAUtilities_cudaCheck_h

// C++ standard headers
#include <iostream>
#include <sstream>
#include <stdexcept>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

namespace {
  inline std::ostringstream formatCudaError(const char* file, int line, const char* cmd, const char* error, const char* message) {
    std::ostringstream out;
    out << "\n";
    out << file << ", line " << line << ":\n";
    out << "cudaCheck(" << cmd << ");\n";
    out << error << ": " << message << "\n";
    return out;
  }

  [[noreturn]] inline void abortOnCudaError(const char* file, int line, const char* cmd, const char* error, const char* message) {
    auto out = formatCudaError(file, line, cmd, error, message);
    throw std::runtime_error(out.str());
  }

  template <typename T>
  [[noreturn]] inline void abortOnCudaError(const char* file, int line, const char* cmd, const char* error, const char* message, T const& description) {
    auto out = formatCudaError(file, line, cmd, error, message);
    out << description << "\n";
    throw std::runtime_error(out.str());
  }

  inline const char* cudaErrorName(CUresult result) {
    const char *error;
    cuGetErrorName(result, &error);
    return error;
  }

  inline const char *cudaErrorString(CUresult result) {
    const char* message;
    cuGetErrorString(result, &message);
    return message;
  }

  inline bool cudaIsSuccess(CUresult result) {
    return result == CUDA_SUCCESS;
  }

  inline const char*cudaErrorName(cudaError_t result) {
    return cudaGetErrorName(result);
  }

  inline const char*cudaErrorString(cudaError_t result) {
    return cudaGetErrorString(result);
  }

  inline bool cudaIsSuccess(cudaError_t result) {
    return result == cudaSuccess;
  }
}  // namespace

template <typename RESULT, typename... Args>
inline bool cudaCheck_(const char* file, int line, const char* cmd, RESULT result, Args&&... args) {
  if (__builtin_expect(cudaIsSuccess(result), true))
    return true;

  const char* error = cudaErrorName(result);
  const char* message = cudaErrorString(result);
  abortOnCudaError(file, line, cmd, error, message, std::forward<Args>(args)...);
  return false;
}

#define cudaCheck(ARG) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG)))

#define cudaCheckVerbose(ARG, DESC) (cudaCheck_(__FILE__, __LINE__, #ARG, (ARG), DESC))

#endif  // HeterogeneousCore_CUDAUtilities_cudaCheck_h
