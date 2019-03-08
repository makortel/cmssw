#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigiErrorsCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

SiPixelDigiErrorsCUDA::SiPixelDigiErrorsCUDA(size_t maxFedWords, PixelFormatterErrors errors, cuda::stream_t<>& stream):
  maxFedWords_h(maxFedWords),
  formatterErrors_h(std::move(errors))
{
  edm::Service<CUDAService> cs;

  error_d = cs->make_device_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  data_d = cs->make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

  cudautils::memsetAsync(data_d, 0x00, maxFedWords, stream);

  auto error_h = cs->make_host_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  GPU::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
  assert(error_h->size() == 0);
  assert(error_h->capacity() == static_cast<int>(maxFedWords));

  cudautils::copyAsync(error_d, error_h, stream);
}

SiPixelDigiErrorsCUDA::HostDataError SiPixelDigiErrorsCUDA::dataErrorToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto error = cs->make_host_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
  auto data = cs->make_host_unique<PixelErrorCompact[]>(maxFedWords_h, stream);

  cudautils::copyAsync(error, error_d, stream);
  cudautils::copyAsync(data, data_d, maxFedWords_h, stream);

  return HostDataError(std::move(error), std::move(data));
}
