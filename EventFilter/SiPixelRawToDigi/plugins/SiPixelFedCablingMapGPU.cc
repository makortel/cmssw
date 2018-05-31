// C++ includes
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>

// CMSSW includes
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLT.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelGainForHLTonGPU.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/CUDAUtilities/interface/CUDAHostAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

// local includes
#include "SiPixelFedCablingMapGPU.h"

void
processGainCalibration(SiPixelGainCalibrationForHLT const & gains, TrackerGeometry const& geom,
                       SiPixelGainForHLTonGPU *gainsOnHost, SiPixelGainForHLTonGPU * & gainsOnGPU,
                       SiPixelGainForHLTonGPU::DecodingStructure * & gainDataOnGPU, cuda::stream_t<>& stream) {
  // bizzarre logic (looking for fist strip-det) don't ask
  auto const & dus = geom.detUnits();
  unsigned m_detectors = dus.size();
  for(unsigned int i=1;i<7;++i) {
    if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size() &&
        dus[geom.offsetDU(GeomDetEnumerators::tkDetEnum[i])]->type().isTrackerStrip()) {
      if(geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_detectors) m_detectors = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }

  /*
  std::cout << "caching calibs for " << m_detectors << " pixel detectors of size " << gains.data().size() << std::endl;
  std::cout << "sizes " << sizeof(char) << ' ' << sizeof(uint8_t) << ' ' << sizeof(SiPixelGainForHLTonGPU::DecodingStructure) << std::endl;
  */

  SiPixelGainForHLTonGPU * gg = gainsOnHost;

  assert(nullptr==gainDataOnGPU);
  cudaCheck(cudaMalloc((void**) & gainDataOnGPU, gains.data().size())); // TODO: this could be changed to cuda::memory::device::unique_ptr<>
  // gains.data().data() is used also for non-GPU code, we cannot allocate it on aligned and write-combined memory
  cudaCheck(cudaMemcpyAsync(gainDataOnGPU, gains.data().data(), gains.data().size(), cudaMemcpyDefault, stream.id()));

  gg->v_pedestals = gainDataOnGPU;

  // do not read back from the (possibly write-combined) memory buffer
  auto minPed  = gains.getPedLow();
  auto maxPed  = gains.getPedHigh();
  auto minGain = gains.getGainLow();
  auto maxGain = gains.getGainHigh();
  auto nBinsToUseForEncoding = 253;

  // we will simplify later (not everything is needed....)
  gg->minPed_ = minPed;
  gg->maxPed_ = maxPed;
  gg->minGain_= minGain;
  gg->maxGain_= maxGain;

  gg->numberOfRowsAveragedOver_ = 80;
  gg->nBinsToUseForEncoding_    = nBinsToUseForEncoding;
  gg->deadFlag_                 = 255;
  gg->noisyFlag_                = 254;

  gg->pedPrecision  = static_cast<float>(maxPed - minPed) / nBinsToUseForEncoding;
  gg->gainPrecision = static_cast<float>(maxGain - minGain) / nBinsToUseForEncoding;

  /*
  std::cout << "precisions g " << gg->pedPrecision << ' ' << gg->gainPrecision << std::endl;
  */

  // fill the index map
  auto const & ind = gains.getIndexes();  
  /*
  std::cout << ind.size() << " " << m_detectors << std::endl;
  */

  for (auto i=0U; i<m_detectors; ++i) {
    auto p = std::lower_bound(ind.begin(),ind.end(),dus[i]->geographicalId().rawId(),SiPixelGainCalibrationForHLT::StrictWeakOrdering());
    assert (p!=ind.end() && p->detid==dus[i]->geographicalId());
    assert(p->iend<=gains.data().size());
    assert(p->iend>=p->ibegin);
    assert(0==p->ibegin%2);
    assert(0==p->iend%2);
    assert(p->ibegin!=p->iend);
    assert(p->ncols>0);
    gg->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(p->ibegin,p->iend), p->ncols);
    // if (ind[i].detid!=dus[i]->geographicalId()) std::cout << ind[i].detid<<"!="<<dus[i]->geographicalId() << std::endl;
    // gg->rangeAndCols[i] = std::make_pair(SiPixelGainForHLTonGPU::Range(ind[i].ibegin,ind[i].iend), ind[i].ncols);
  }

  cudaCheck(cudaMemcpyAsync(gainsOnGPU, gg, sizeof(SiPixelGainForHLTonGPU), cudaMemcpyDefault, stream.id()));
}
