#ifndef DataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h
#define DataFormats_SiPixelDigi_interface_SiPixelDigisSoA_h

#include "DataFormats/SiPixelDigi/interface/PixelErrors.h"

#include <cstdint>
#include <vector>

class SiPixelDigisSoA {
public:
  SiPixelDigisSoA() = default;
  explicit SiPixelDigisSoA(size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus);
  explicit SiPixelDigisSoA(size_t nDigis, const uint32_t *pdigi, const uint32_t *rawIdArr, const uint16_t *adc, const int32_t *clus,
                           size_t nErrors, const PixelErrorCompact *error, const PixelFormatterErrors *err);
  ~SiPixelDigisSoA() = default;

  auto size() const { return pdigi_.size(); }
  auto errorSize() const { return error_.size(); }

  bool hasError() const { return hasError_; }
  const PixelFormatterErrors *formatterErrors() const { return formatterErrors_; }

  uint32_t pdigi(size_t i) const { return pdigi_[i]; }
  uint32_t rawIdArr(size_t i) const { return rawIdArr_[i]; }
  uint16_t adc(size_t i) const { return adc_[i]; }
  int32_t clus(size_t i) const { return clus_[i]; }
  const PixelErrorCompact& error(size_t i) const { return error_[i]; }
  
  const std::vector<uint32_t>& pdigiVector() const { return pdigi_; }
  const std::vector<uint32_t>& rawIdArrVector() const { return rawIdArr_; }
  const std::vector<uint16_t>& adcVector() const { return adc_; }
  const std::vector<int32_t>& clusVector() const { return clus_; }
  const std::vector<PixelErrorCompact>& errorVector() const { return error_; }
  
private:
  std::vector<uint32_t> pdigi_;
  std::vector<uint32_t> rawIdArr_;
  std::vector<uint16_t> adc_;
  std::vector<int32_t> clus_;

  std::vector<PixelErrorCompact> error_;
  const PixelFormatterErrors *formatterErrors_ = nullptr;
  bool hasError_ = false;
};

#endif
