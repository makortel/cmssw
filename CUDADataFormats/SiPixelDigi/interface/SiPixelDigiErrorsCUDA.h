#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigiErrorsCUDA_h

#include "DataFormats/SiPixelDigi/interface/PixelErrors.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/GPUSimpleVector.h"

#include <cuda/api_wrappers.h>

class SiPixelDigiErrorsCUDA {
public:
  SiPixelDigiErrorsCUDA() = default;
  explicit SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cuda::stream_t<>& stream);
  ~SiPixelDigiErrorsCUDA() = default;

  SiPixelDigiErrorsCUDA(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA& operator=(const SiPixelDigiErrorsCUDA&) = delete;
  SiPixelDigiErrorsCUDA(SiPixelDigiErrorsCUDA&&) = default;
  SiPixelDigiErrorsCUDA& operator=(SiPixelDigiErrorsCUDA&&) = default;

  const PixelFormatterErrors& formatterErrors() const { return formatterErrors_h; }

  GPU::SimpleVector<PixelErrorCompact> *error() { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const *error() const { return error_d.get(); }
  GPU::SimpleVector<PixelErrorCompact> const *c_error() const { return error_d.get(); }

  // Note: the HostDataError.first.set_data(HostDataError.second) must
  // be called explicitly after synchronizing
  using HostDataError = std::pair<cudautils::host::unique_ptr<GPU::SimpleVector<PixelErrorCompact>>,
                                  cudautils::host::unique_ptr<PixelErrorCompact[]>>;
  HostDataError dataErrorToHostAsync(cuda::stream_t<>& stream) const;

private:
  cudautils::device::unique_ptr<PixelErrorCompact[]> data_d;
  cudautils::device::unique_ptr<GPU::SimpleVector<PixelErrorCompact>> error_d;
  size_t maxFedWords_h;
  PixelFormatterErrors formatterErrors_h;
};

#endif
