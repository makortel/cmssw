#ifndef CondFormats_DataRecord_SiPixelDynamicInefficiencyMixingRcd_h
#define CondFormats_DataRecord_SiPixelDynamicInefficiencyMixingRcd_h

#include "CondFormats/DataRecord/interface/MixingRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelDynamicInefficiencyRcd.h"
#include "FWCore/Framework/interface/DependentRecordImplementation.h"

class SiPixelDynamicInefficiencyMixingRcd
    : public edm::eventsetup::DependentRecordImplementation<SiPixelDynamicInefficiencyMixingRcd,
                                                            edm::mpl::Vector<MixingRcd, SiPixelDynamicInefficiencyRcd>> {
};

#endif
