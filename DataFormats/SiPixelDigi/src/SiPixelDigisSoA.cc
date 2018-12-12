#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"

#include <algorithm>
#include <cassert>

SiPixelDigisSoA::SiPixelDigisSoA(size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus):
  pdigi_(pdigi, pdigi+nDigis),
  rawIdArr_(rawIdArr, rawIdArr+nDigis),
  adc_(adc, adc+nDigis),
  clus_(clus, clus+nDigis)
{
  assert(pdigi_.size() == nDigis);
}

SiPixelDigisSoA::SiPixelDigisSoA(size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus,
                                 size_t nErrors, const PixelErrorCompact *error, const PixelFormatterErrors *err):
  SiPixelDigisSoA(nDigis, pdigi, rawIdArr, adc, clus)
{
  error_.resize(nErrors);
  std::copy(error, error+nErrors, error_.begin());
  formatterErrors_ = err;
  hasError_ = true;
}
