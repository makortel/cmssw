#include "CUDADataFormats/SiPixelDigi/interface/SiPixelDigisCUDA.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

SiPixelDigisCUDA::SiPixelDigisCUDA(size_t nelements, cuda::stream_t<>& stream) {
  edm::Service<CUDAService> cs;

  xx_d              = cs->make_unique<uint16_t[]>(nelements, stream);
  yy_d              = cs->make_unique<uint16_t[]>(nelements, stream);
  adc_d             = cs->make_unique<uint16_t[]>(nelements, stream);
  moduleInd_d       = cs->make_unique<uint16_t[]>(nelements, stream);
}
