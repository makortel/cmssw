#include "RecoTracker/TkHitPairs/interface/IntermediateHitDoublets.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <vector>

namespace RecoTracker_TkTrackingRegions {
  struct dictionary {
    /*
    std::vector<IntermediateHitDoublets> vihd;
    edm::Wrapper<std::vector<IntermediateHitDoublets> > wvihd;
    */
    IntermediateHitDoublets ihd;
    edm::Wrapper<IntermediateHitDoublets> wihd;
  };
}
