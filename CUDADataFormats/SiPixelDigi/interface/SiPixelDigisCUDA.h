#ifndef CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h
#define CUDADataFormats_SiPixelDigi_interface_SiPixelDigisCUDA_h

#include "CUDADataFormats/Common/interface/device_unique_ptr.h"

#include <cuda/api_wrappers.h>

class SiPixelDigisCUDA {
public:
  SiPixelDigisCUDA() = default;
  explicit SiPixelDigisCUDA(size_t nelements, cuda::stream_t<>& stream);
  ~SiPixelDigisCUDA() = default;

  SiPixelDigisCUDA(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA& operator=(const SiPixelDigisCUDA&) = delete;
  SiPixelDigisCUDA(SiPixelDigisCUDA&&) = default;
  SiPixelDigisCUDA& operator=(SiPixelDigisCUDA&&) = default;

  uint16_t * __restrict__ xx() { return xx_d.get(); }
  uint16_t * __restrict__ yy() { return yy_d.get(); }
  uint16_t * __restrict__ adc() { return adc_d.get(); }
  uint16_t * __restrict__ moduleInd() { return moduleInd_d.get(); }

  uint16_t const * __restrict__ xx() const { return xx_d.get(); }
  uint16_t const * __restrict__ yy() const { return yy_d.get(); }
  uint16_t const * __restrict__ adc() const { return adc_d.get(); }
  uint16_t const * __restrict__ moduleInd() const { return moduleInd_d.get(); }

  uint16_t const * __restrict__ c_xx() const { return xx_d.get(); }
  uint16_t const * __restrict__ c_yy() const { return yy_d.get(); }
  uint16_t const * __restrict__ c_adc() const { return adc_d.get(); }
  uint16_t const * __restrict__ c_moduleInd() const { return moduleInd_d.get(); }

  struct DeviceConstView {
    uint16_t const * xx;
    uint16_t const * yy;
    uint16_t const * adc;
    uint16_t const * moduleInd;
  };

  DeviceConstView view() const { return DeviceConstView{xx_d.get(), yy_d.get(), adc_d.get(), moduleInd_d.get()}; }

private:
  edm::cuda::device::unique_ptr<uint16_t[]> xx_d;        // local coordinates of each pixel
  edm::cuda::device::unique_ptr<uint16_t[]> yy_d;        //
  edm::cuda::device::unique_ptr<uint16_t[]> adc_d;       // ADC of each pixel
  edm::cuda::device::unique_ptr<uint16_t[]> moduleInd_d; // module id of each pixel
};

#endif
