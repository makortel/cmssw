#include <cuda_runtime.h>

int main(void) {
  int n;
  auto status = cudaGetDeviceCount(&n);
  return status == cudaSuccess ? 0 : 1;
}
