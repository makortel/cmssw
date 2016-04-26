#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace RecoTracker_TkTrackingRegions {
  struct dictionary {
    edm::Wrapper<std::vector<std::unique_ptr<TrackingRegion> > > wvutr;
  };
}
