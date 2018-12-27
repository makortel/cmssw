#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/copyAsync.h"
#include "HeterogeneousCore/CUDAUtilities/interface/memsetAsync.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t maxFedWords, bool includeErrors, cuda::stream_t<>& stream):
  hasErrors_h(includeErrors)
{
  edm::Service<CUDAService> cs;

  xx_d              = cs->make_device_unique<uint16_t[]>(maxFedWords, stream);
  yy_d              = cs->make_device_unique<uint16_t[]>(maxFedWords, stream);
  adc_d             = cs->make_device_unique<uint16_t[]>(maxFedWords, stream);
  moduleInd_d       = cs->make_device_unique<uint16_t[]>(maxFedWords, stream);
  clus_d            = cs->make_device_unique< int32_t[]>(maxFedWords, stream);

  pdigi_d           = cs->make_device_unique<uint32_t[]>(maxFedWords, stream);
  rawIdArr_d        = cs->make_device_unique<uint32_t[]>(maxFedWords, stream);

  auto view = cs->make_host_unique<DeviceConstView>(stream);
  view->xx_ = xx_d.get();
  view->yy_ = yy_d.get();
  view->adc_ = adc_d.get();
  view->moduleInd_ = moduleInd_d.get();
  view->clus_ = clus_d.get();

  view_d = cs->make_device_unique<DeviceConstView>(stream);
  cudautils::copyAsync(view_d, view, stream);

  if(includeErrors) {
    error_d = cs->make_device_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
    data_d = cs->make_device_unique<PixelErrorCompact[]>(maxFedWords, stream);

    cudautils::memsetAsync(data_d, 0x00, maxFedWords, stream);

    error_h = cs->make_host_unique<GPU::SimpleVector<PixelErrorCompact>>(stream);
    GPU::make_SimpleVector(error_h.get(), maxFedWords, data_d.get());
    assert(error_h->size() == 0);
    assert(error_h->capacity() == static_cast<int>(maxFedWords));

    cudautils::copyAsync(error_d, error_h, stream);
  }
}

cudautils::host::unique_ptr<uint16_t[]> SiPixelDigisCUDA::adcToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<uint16_t[]>(nDigis(), stream);
  cudautils::copyAsync(ret, adc_d, nDigis(), stream);
  return ret;
}

cudautils::host::unique_ptr<int32_t[]> SiPixelDigisCUDA::clusToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<int32_t[]>(nDigis(), stream);
  cudautils::copyAsync(ret, clus_d, nDigis(), stream);
  return ret;
}

cudautils::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::pdigiToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<uint32_t[]>(nDigis(), stream);
  cudautils::copyAsync(ret, pdigi_d, nDigis(), stream);
  return ret;
}

cudautils::host::unique_ptr<uint32_t[]> SiPixelDigisCUDA::rawIdArrToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  auto ret = cs->make_host_unique<uint32_t[]>(nDigis(), stream);
  cudautils::copyAsync(ret, rawIdArr_d, nDigis(), stream);
  return ret;
}

void SiPixelDigisCUDA::copyErrorToHostAsync(cuda::stream_t<>& stream) {
  cudautils::copyAsync(error_h, error_d, stream);
}


SiPixelDigisCUDA::HostDataError SiPixelDigisCUDA::dataErrorToHostAsync(cuda::stream_t<>& stream) const {
  edm::Service<CUDAService> cs;
  // On one hand size() could be sufficient. On the other hand, if
  // someone copies the SimpleVector<>, (s)he might expect the data
  // buffer to actually have space for capacity() elements.
  auto data = cs->make_host_unique<PixelErrorCompact[]>(error_h->capacity(), stream);

  // but transfer only the required amount
  if(error_h->size() > 0) {
    cudautils::copyAsync(data, data_d, error_h->size(), stream);
  }
  error_h->set_data(data.get());
  return HostDataError(std::move(data), error_h.get());
}
  
